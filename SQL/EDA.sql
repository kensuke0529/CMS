#1:
-- Write a SQL query to list the top 10 U.S. states (by abbreviation) that received the highest average Medicare payment per discharge in the year 2022.

select 
	SUM(Avg_Mdcr_Pymt_Amt * Tot_Dschrgs) as total_medicare_payment,
	Rndrng_Prvdr_State_Abrvtn
from medicare
where year = 2022
group by Rndrng_Prvdr_State_Abrvtn
order by total_medicare_payment desc
limit 10;

#2:
-- Find the top 10 DRG codes with the highest average submitted charge per discharge in 2022, 
-- but only for DRGs that had more than 1000 total discharges.

select 	
	DRG_Cd, 
    sum(Avg_Submtd_Cvrd_Chrg * Tot_Dschrgs) as total_submitted_charges,
	sum(Tot_Dschrgs) as  total_discharges,
	round(sum(Avg_Submtd_Cvrd_Chrg * Tot_Dschrgs) / sum(Tot_Dschrgs), 2) as  avg_submitted_charge_per_discharge
from medicare
where year = 2022
group by DRG_Cd
having SUM(Tot_Dschrgs) >= 1000
order by avg_submitted_charge_per_discharge desc
limit 10;


#3: Categorize DRGs by Cost Level
-- Create a query that lists each DRG_Cd and categorizes it into three cost buckets based on the average submitted charge per discharge for the year 2022:
-- High Cost: avg submitted charge per discharge > $50,000
-- Medium Cost: avg submitted charge per discharge between $20,000 and $50,000
-- Low Cost: avg submitted charge per discharge < $20,000

select 
	cost_category, 
	count(cost_category) as category_count,
    round(avg(avg_submitted_charge_per_discharge),2)  as category_avg
from(
	select 
		DRG_Cd,
		round(sum(Avg_Submtd_Cvrd_Chrg * Tot_Dschrgs) / sum(Tot_Dschrgs), 2) as  avg_submitted_charge_per_discharge,
		case
			when round(sum(Avg_Submtd_Cvrd_Chrg * Tot_Dschrgs) / sum(Tot_Dschrgs), 2) > 50000 then 'High'
			when round(sum(Avg_Submtd_Cvrd_Chrg * Tot_Dschrgs) / sum(Tot_Dschrgs), 2) between 20000 and 50000 then 'Medium'
			else  'Low' 
		end as cost_category
	from medicare 
	group by DRG_Cd
	order by avg_submitted_charge_per_discharge desc) as sub
group by cost_category; 


#4: Rank DRGs by Average Medicare Payment within Each Cost Category

-- For the year 2022, write a query that:
-- Calculates the weighted average Medicare payment per discharge for each DRG_Cd
-- Categorizes each DRG into High, Medium, or Low cost buckets using the same thresholds you used before
-- Assigns a rank to each DRG within its cost category, based on the average Medicare payment per discharge, with 1 being the highest

select * 
from (
	select 
		DRG_Cd,
		cost_category,
		avg_medicare_amount_per_discharge,
		rank () over (partition by cost_category order by avg_medicare_amount_per_discharge desc) as rank_in_category
	from 
		(select
			DRG_Cd,
			round(sum(Avg_Mdcr_Pymt_Amt * Tot_Dschrgs) / sum(Tot_Dschrgs), 2) as  avg_medicare_amount_per_discharge,
			case
				when round(sum(Avg_Mdcr_Pymt_Amt * Tot_Dschrgs) / sum(Tot_Dschrgs), 2) > 50000 then 'High'
				when round(sum(Avg_Mdcr_Pymt_Amt * Tot_Dschrgs) / sum(Tot_Dschrgs), 2) between 20000 and 50000 then 'Medium'
				else  'Low' 
			end as cost_category
		from medicare
		where year = 2022
		group by DRG_Cd) as sub
) as ranked
where rank_in_category = 1;

 -- Altenative solutions using CTE
with drg_cat as (
	select
		DRG_Cd,
		round(sum(Avg_Mdcr_Pymt_Amt * Tot_Dschrgs) / sum(Tot_Dschrgs), 2) as  avg_medicare_amount_per_discharge,
		case
			when round(sum(Avg_Mdcr_Pymt_Amt * Tot_Dschrgs) / sum(Tot_Dschrgs), 2) > 50000 then 'High'
			when round(sum(Avg_Mdcr_Pymt_Amt * Tot_Dschrgs) / sum(Tot_Dschrgs), 2) between 20000 and 50000 then 'Medium'
			else  'Low' 
		end as cost_category
	from medicare
	where year = 2022
	group by DRG_Cd
), 
ranked_drg as (
	select	 
		DRG_Cd,
		cost_category,
		avg_medicare_amount_per_discharge,
		rank () over (partition by cost_category order by avg_medicare_amount_per_discharge desc) as rank_in_category
	from drg_cat
)
select * 
from ranked_drg
where rank_in_category = 1;

#5: Track Change in Average Medicare Payment Over Time (Using LAG())
-- Scenario:
-- CMS wants to understand how the average Medicare payment per discharge for each DRG_Cd has changed year-over-year. Your job is to calculate:
-- The average Medicare payment per discharge for each DRG_Cd in each year
-- The change in payment compared to the previous year
-- Rank DRG codes that had the largest increases in 2022


with prep_table as (
	select
		DRG_Cd,
		year,
		round(sum(Avg_Mdcr_Pymt_Amt * Tot_Dschrgs) / sum(Tot_Dschrgs), 2) as  avg_medicare_amount_per_discharge
	from medicare
	group by DRG_Cd, year
),
sub1 as (
	select
		DRG_Cd,
		year, 
		avg_medicare_amount_per_discharge,
		lag (avg_medicare_amount_per_discharge) over (partition by DRG_Cd order by year) as prev_amount
	from prep_table
)
select  DRG_Cd, year, avg_medicare_amount_per_discharge,prev_amount,increase_percentage, ranking
from (
	select DRG_Cd, year, avg_medicare_amount_per_discharge,prev_amount,
			round((avg_medicare_amount_per_discharge / prev_amount - 1) * 100 ,2) as increase_percentage,
			rank () over (order by round((avg_medicare_amount_per_discharge / prev_amount - 1) * 100 ,2) desc) as ranking
	from sub1
	where year = 2022
) as table1 
where ranking < 10;

 #6: Compute Rolling 3-Year Average Medicare Payment Using WINDOW
-- Scenario:
-- CMS wants to smooth out year-to-year fluctuations in DRG payment by calculating a rolling 3-year average Medicare payment per discharge for each DRG_Cd.
-- Your goal is to compute, for each DRG and each year:
-- The average Medicare payment per discharge
-- The 3-year rolling average, including the current year and 2 years before it
-- Only include rows where there are at least 3 years of data available (to make a full rolling window)


with drg_yearly as (
select 
	DRG_Cd,
    year,
    round(sum(Avg_Mdcr_Pymt_Amt * Tot_Dschrgs) / sum(Tot_Dschrgs), 2) as  avg_medicare_amount_per_discharge
from medicare
group by DRG_Cd, year
),
rolling_avg as (
	select 
		DRG_Cd, 
        year,
		avg_medicare_amount_per_discharge,
        round(
			avg(avg_medicare_amount_per_discharge) 
			over (partition by DRG_Cd order by year rows between 2 preceding and current row)
        ,2) as rolling_3years
	from drg_yearly 
)
select DRG_Cd, year, avg_medicare_amount_per_discharge, rolling_3years, round(avg_medicare_amount_per_discharge - rolling_3years,2) as gap
from rolling_avg
where rolling_3years is not null
order by DRG_Cd;


#7: Identify the Most Profitable DRG Codes by State (2022)
-- Scenario:
-- You're working with Medicare data and need to identify, per state, the top DRG code by total Medicare payment in 2022.

with ranking as (
select DRG_cd, Rndrng_Prvdr_State_Abrvtn, SUM(Avg_Mdcr_Pymt_Amt * Tot_Dschrgs) AS total_medicare, 
	rank () over (partition by Rndrng_Prvdr_State_Abrvtn order by sum(Avg_Mdcr_Pymt_Amt) desc) as state_rank 
from medicare 
where year = 2022
group by DRG_cd, Rndrng_Prvdr_State_Abrvtn
)
select DRG_cd, Rndrng_Prvdr_State_Abrvtn,total_medicare, state_rank
from ranking 
where state_rank <= 3;


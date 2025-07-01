-- insert into providers table 
insert into providers(provider_id, name, city, street, fips, zip_code, state, ruca, ruca_category) 
select * from (
	with ruca_mode as (
		select
			Rndrng_Prvdr_CCN,
			Rndrng_Prvdr_RUCA,
			RUCA_category,
			count(*) as cnt,
			row_number() over (partition BY Rndrng_Prvdr_CCN order by ciybt(*) desc) as rn
	  from medicare
	  group by Rndrng_Prvdr_CCN, Rndrng_Prvdr_RUCA, RUCA_category
	),
	sub_table as (
		select 
		  m.Rndrng_Prvdr_CCN,
		  MAX(m.Rndrng_Prvdr_Org_Name) as name,
		  MAX(m.Rndrng_Prvdr_City) as city,
		  MAX(m.Rndrng_Prvdr_St) as street,
		  MAX(m.Rndrng_Prvdr_State_FIPS) as fips,
		  MAX(m.Rndrng_Prvdr_Zip5) as zip_code,
		  MAX(m.Rndrng_Prvdr_State_Abrvtn) as state,
		  r.Rndrng_Prvdr_RUCA,
		  r.RUCA_category
		from medicare m
		join ruca_mode r on m.Rndrng_Prvdr_CCN = r.Rndrng_Prvdr_CCN
		where r.rn = 1
		group by m.Rndrng_Prvdr_CCN, r.Rndrng_Prvdr_RUCA, r.RUCA_category
	) 
    select * from sub_table
) as cleaned; 


-- insert into drgs table 
insert into drgs (drg_code, year, drg_description)
select distinct 
  DRG_Cd,
  year,
  DRG_Desc
from medicare;


-- insert payment 
insert into payments (provider_id, drg_code, year, tot_discharge, avg_submtd_cvrd_chrg, tot_payment, avg_medicare_payment)
select distinct 
	Rndrng_Prvdr_CCN,
    DRG_Cd,
    year, 
    Tot_dschrgs,
    Avg_Submtd_Cvrd_Chrg,
    Avg_Tot_Pymt_Amt,
    Avg_Mdcr_Pymt_Amt
from medicare; 


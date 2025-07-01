-- For each product, list its name and total quantity ordered. Products should be listed in ascending order of the product name.

select product_name, sum(quantity) as total_ordered
from product as p 
inner join order_line as l on p.product_id = l.product_id
group by product_name
order by product_name asc;

-- For each customer in Texas, list their name, city and total dollars spent.  Sort the customers from highest revenue to lowest revenue. 
select customer_name, city, sum(quantity * product_price) as revenue
from product as p 
inner join order_line as l on p.product_id = l.product_id
inner join order_header as h on h.order_id = l.order_id
inner join customer as c on c.customer_id  = h.customer_id
where upper(c.state_province) = 'TEXAS' 
group by customer_name, city
order by revenue desc;


-- For each product, list its ID, name and total revenue for 2023. Products should be listed in ascending order of the product name.
select p.product_id, product_name, sum(quantity * product_price) as revenue
from product as p 
inner join order_line as l on p.product_id = l.product_id
inner join order_header as h on h.order_id = l.order_id
inner join customer as c on c.customer_id  = h.customer_id
where h.order_year = 2023
group by p.product_id, product_name
order by p.product_name asc;


-- List the email and name for all customers who have placed 3 or more orders. Each customer name should appear exactly once. Customer emails should be sorted in descending alphabetical order.
select distinct customer_name, email 
from customer c 
join order_header as h on c.customer_id = h.customer_id
group by c.customer_id, c.customer_name, email
having count(c.customer_id) >=3
order by email desc;

-- Implement the previous query using a subquery and IN adding the requirement that the customers’ orders have been placed on or after Oct 5, 2022.
select customer_name, email
from customer as c 
where customer_id in (
	select h.customer_id 
    from order_header as h 
    group by customer_id 
    having count(*) or max(h.order_date) > '2022-10-05'
)
order by email desc; 

-- For each city in California, list the name of the city and number of customers from that city who have purchased Clothing.  Filter out any customers who have not ordered clothing.  Cities are sorted by the number of customers, descending.
select c.city, count(distinct c.customer_id) as total_customer 
from product as p 
inner join order_line as l on p.product_id = l.product_id
inner join order_header as h on h.order_id = l.order_id
inner join customer as c on c.customer_id  = h.customer_id
where product_line = 'Clothing' and upper(c.state_province) = 'CALIFORNIA'
group by city
order by total_customer desc; 

-- Implement the previous using a subquery and IN.
select city, count(distinct customer_id) as total_customer 
from customer 
where customer_id in (
	select customer_id 
    from order_header as h
	join order_line as l on h.order_id = l.order_id
    join product as p on p.product_id = l.product_id
	where p.product_line = 'Clothing' 
) and upper(state_province) = 'CALIFORNIA'
group by city
order by total_customer desc; 

-- List the ID for all products, which have NOT been ordered on Dec 5, 2023 or before. Sort your results by the product id in ascending order.  Use EXCEPT for this query.
select product_id 
from (
	select product_id 
	from order_line
	except
	select product_id 
	from order_line as l 
	join order_header as h on l.order_id = h.order_id
	where h.order_date <= '2023-12-05' 
	) as sub
order by product_id desc;

-- List the ID for all California customers, who have placed one or more orders in November 2023 or after. Sort the results by the customer id in ascending order.  Use Intersect for this query.  Make sure to look for alternate spellings for California, like "CA"
select customer_id 
from customer 
where upper(state_province) in ('CALIFORNIA' , 'CA')
intersect 
select c.customer_id
from customer as c 
join order_header as h on c.customer_id = h.customer_id
where  h.order_date >= '2023-11-01';

-- Implement the previous query using a subquery and IN.
select customer_id 
from customer 
where customer_id in (
	select c.customer_id
	from customer as c 
	join order_header as h on c.customer_id = h.customer_id
	where  h.order_date >= '2023-11-01' and upper(state_province) in ('CALIFORNIA' , 'CA')
);

-- List the IDs for all California customers along with all customers (regardless where they are from) who have placed one or more order(s) before December 2022. Sort the results by the customer id in descending order.  Use union for this query.
select customer_id 
from customer 
where customer_id in(
	select c.customer_id
    from customer as c 
    join order_header as h on c.customer_id = h.customer_id
    group by customer_id
    having max(h.order_date) > '2022-12-31'
) and upper(state_province) in ('CALIFORNIA' , 'CA');
 
-- List the product ID, product name and total quantity ordered for all products with total quantity ordered of less than 50.
select product_id, product_name,  total_quantity
from (
	select p.product_id, product_name, sum(l.quantity) as total_quantity
	from product as p 
	join order_line as l on p.product_id = l.product_id
	group by p.product_id, p.product_name
) as sub
where total_quantity < 50; 

-- List the product ID, product name  and total quantity ordered for all products with total quantity ordered greater than 3 and were placed by Illinois customers.  Make sure to look for alternative spellings for Illinois state.
select product_id, product_name, total_quantity
from (
	select p.product_id, product_name, sum(l.quantity) as total_quantity
    from product as p 
	inner join order_line as l on p.product_id = l.product_id
	inner join order_header as h on h.order_id = l.order_id
	inner join customer as c on c.customer_id  = h.customer_id
    where upper(c.state_province) in ('ILLINOIS', 'IL')
    group by p.product_id, product_name
    ) as sub
where total_quantity > 3;
    

-- List the customer name and city for all customers who have purchased products that are no longer active status.  
select customer_name, city 
from customer
where customer_id in(
	select c.customer_id
	from product as p 
	inner join order_line as l on p.product_id = l.product_id
	inner join order_header as h on h.order_id = l.order_id
	inner join customer as c on c.customer_id  = h.customer_id
	where product_status != 'active'
);
-- List the ID, name, and price for all products with less than or equal to the average product price.
with avg_table as(
	select avg(product_price) as avg_price
    from product 
)
select product_id, product_name, product_price
from product
where product_price <= (select avg_price from avg_table)

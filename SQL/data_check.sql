select  
Rndrng_Prvdr_RUCA,
RUCA_category,
	count(Rndrng_Prvdr_RUCA)
from medicare
where Rndrng_Prvdr_CCN = 10012
group by Rndrng_Prvdr_RUCA,RUCA_category; 

SET SQL_SAFE_UPDATES = 0;

UPDATE medicare
SET
  Rndrng_Prvdr_RUCA = '1',
  RUCA_category = 'metro_core'
WHERE Rndrng_Prvdr_RUCA = '2';

-- # Rndrng_Prvdr_RUCA, Rndrng_Prvdr_RUCA_desc, count(Rndrng_Prvdr_RUCA)
-- '1', 'Metropolitan area core: primary flow within an urbanized area of 50,000 and greater', '212'
-- '2', 'Metropolitan area high commuting: primary flow 30% or more to a urbanized area of 50,000 and greater', '91'

select distinct 
	Rndrng_Prvdr_CCN, 
    Rndrng_Prvdr_Org_Name, 
    Rndrng_Prvdr_City, 
    Rndrng_Prvdr_St, 
    Rndrng_Prvdr_State_FIPS, 
    Rndrng_Prvdr_Zip5, 
    Rndrng_Prvdr_State_Abrvtn, 
    Rndrng_Prvdr_RUCA, 
    RUCA_category
from medicare 
where Rndrng_Prvdr_CCN = 10001;


select  
Rndrng_Prvdr_RUCA,
RUCA_category,
year,
	count(Rndrng_Prvdr_RUCA)
from medicare
where Rndrng_Prvdr_CCN = 10012
group by Rndrng_Prvdr_RUCA,RUCA_category, year; 
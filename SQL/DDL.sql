create table providers( 
	provider_id int(6) primary key, 
    name varchar(100),
    city varchar(50),
    street varchar (100), 
    fips varchar (10),
    zip_code char(5),
    state varchar(2),
    ruca varchar(10),
    ruca_category varchar(50)
);

create table drgs(
	drg_code int primary key, 
    drg_description varchar (100)
);

ALTER TABLE drgs
ADD COLUMN year INT;

ALTER TABLE drgs
DROP PRIMARY KEY;
ALTER TABLE drgs ADD PRIMARY KEY (drg_code, year);


create table payments(
	provider_id int(6), 
    drg_code int, 
    year int,
    tot_discharge int,
    avg_submtd_cvrd_chrg decimal (25,5),
    tot_payment decimal (25,5),
    avg_medicare_payment  decimal (25,5),
    
    primary key (provider_id, drg_code, year),
    foreign key (provider_id) references providers(provider_id),
    foreign key (drg_code,year) references drgs(drg_code,year) 
);


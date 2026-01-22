--  Cleaned and validated telecom customer data (3K+ rows) using SQL, handling duplicates, nulls, outliers, and logical inconsistencies
select * from ctable;

create  table ch1
like ctable;

select * from ch1;

insert into ch1
select * from ctable;

select * from ch1;

 -- Assumption-based deduplication due to missing customer_id
DELETE c1
FROM ch1 c1
JOIN (
SELECT State,`Account length`,`Area code`,
	ROW_NUMBER() OVER (PARTITION BY State, `Account length`, `Area code`ORDER BY `Account length`) AS rn
    FROM ch1
) c2
ON  c1.State = c2.State
AND c1.`Account length` = c2.`Account length`
AND c1.`Area code` = c2.`Area code`
WHERE c2.rn > 1;



-- it is used to check that is there any duplicate values if 0 no duplicate values
select  State,
    `Account length`,
    `Area code`,
    `International plan`,
    `Voice mail plan`,
    `Number vmail messages`,
    `Total day minutes`,
    `Total day calls`,
    `Total day charge`,
    `Total eve minutes`,
    `Total eve calls`,
    `Total eve charge`,
    `Total night minutes`,
    `Total night calls`,
    `Total night charge`,
    `Total intl minutes`,
    `Total intl calls`,
    `Total intl charge`,
    `Customer service calls`,
    Churn,count(*) from ch1 group by State,
    `Account length`,
    `Area code`,
    `International plan`,
    `Voice mail plan`,
    `Number vmail messages`,
    `Total day minutes`,
    `Total day calls`,
    `Total day charge`,
    `Total eve minutes`,
    `Total eve calls`,
    `Total eve charge`,
    `Total night minutes`,
    `Total night calls`,
    `Total night charge`,
    `Total intl minutes`,
    `Total intl calls`,
    `Total intl charge`,
    `Customer service calls`,
    Churn having count(*) > 1;

-- for these criteria we have duplicate but for further column names we don't have duplicate value
-- that is mentioned in above query
select * from ch1 where state = 'OH' and `Account length` = 84 and `Area code`=408;


-- for exact column names
SHOW COLUMNS FROM ch1;

-- if any null values exist 
SELECT
    SUM(CASE WHEN State IS NULL THEN 1 ELSE 0 END) AS state_nulls,
    SUM(CASE WHEN `Account length`        IS NULL THEN 1 ELSE 0 END) AS tenure_nulls,
    sum(case when `Number vmail messages` is null then 1 else 0 end) as vmail_message_null,
    SUM(CASE WHEN `Total day minutes`     IS NULL THEN 1 ELSE 0 END) AS day_min_nulls,
    SUM(CASE WHEN `Total day charge`      IS NULL THEN 1 ELSE 0 END) AS day_charge_nulls,
    SUM(CASE WHEN `Total day calls`       IS NULL THEN 1 ELSE 0 END) AS day_call_nulls,
    SUM(CASE WHEN `Total eve minutes`     IS NULL THEN 1 ELSE 0 END) AS eve_min_nulls,
    SUM(CASE WHEN `Total eve calls`       IS NULL THEN 1 ELSE 0 END) AS eve_call_nulls,
    SUM(CASE WHEN `Total eve charge`      IS NULL THEN 1 ELSE 0 END) AS eve_charge_nulls,
    SUM(CASE WHEN `Total night calls`     IS NULL THEN 1 ELSE 0 END) AS night_call_nulls,
    SUM(CASE WHEN `Total night minutes`   IS NULL THEN 1 ELSE 0 END) AS night_call_nulls,
    SUM(CASE WHEN `Total night charge`    IS NULL THEN 1 ELSE 0 END) AS night_charge_nulls,
    SUM(CASE WHEN `Total intl minutes`    IS NULL THEN 1 ELSE 0 END) AS intl_call_nulls,
    SUM(CASE WHEN `Total intl charge`     IS NULL THEN 1 ELSE 0 END) AS intl_charges_nulls,
    SUM(CASE WHEN `Customer service calls` IS NULL THEN 1 ELSE 0 END) AS customer_calls_nulls,
    SUM(CASE WHEN `Churn`                   IS NULL THEN 1 ELSE 0 END) AS churn_nulls
FROM ch1;

-- finding max and min values
select 
  max(`Total day minutes`) ,min(`Total day minutes`), -- for Total Day
  max(`Total day calls`) , min(`Total day calls`),
  max(`Total day charge`),min(`Total day charge`)
from ch1;
 
select
  max(`Total eve minutes`),min(`Total eve minutes`), -- for Total Evening
  max(`Total eve calls`),min(`Total eve calls`),
  max(`Total eve charge`),min(`Total eve charge`) 
from ch1;
 
 select 
  max(`Total night minutes`),min(`Total night minutes`), -- for Total night
  max(`Total night calls`),min(`Total night calls`),
  max(`Total night charge`),min(`Total night charge`)
from ch1;

select
  max(`Total intl minutes`),min(`Total intl minutes`), -- for Total intl
  max(`Total intl calls`),min(`Total intl calls`),
  max(`Total intl charge`),min(`Total intl charge`)
from ch1;

-- abnormal usage exceeding telecom limits
SELECT
  COUNT(*) AS outlier_customers,
  ROUND(COUNT(*)*100.0/ (SELECT COUNT(*) FROM ch1),2) AS pct_affected
FROM ch1
WHERE `Total day minutes` > 350;

-- consistency
select * from ch1;

-- 1.Flag data inconsistency instead of ignoring it
select  count(*) as total_customer , case 
when `International plan` = 'No' and `Total intl minutes` > 0 then 1 else 0
end as data_issue from ch1 group by data_issue ;

-- 2.Voice mail is no but number vmail message is more than 0
select * from ch1 
where `Voice mail plan` = 'No' and `Number vmail messages` > 0; -- no rows

SELECT `Customer service calls`, COUNT(*) AS customers
FROM ch1
GROUP BY `Customer service calls`
ORDER BY `Customer service calls`;
-- based on the above query we have kept threshold value as 8
select * from ch1 where `Customer Service calls` > 8;

-- outliers
SELECT
    t.`Total day minutes` AS p99_day
FROM (
    SELECT
        `Total day minutes`,
        ROW_NUMBER() OVER (ORDER BY `Total day minutes`) AS rn,
        COUNT(*) OVER () AS total_rows
    FROM ch1
) t
WHERE rn = FLOOR(0.99 * total_rows);

-- categorical data
SELECT DISTINCT `International plan` FROM ch1;
SELECT DISTINCT `Voice mail plan` FROM ch1;
SELECT DISTINCT Churn FROM ch1;
-- checking if there is any other than yes,no or true,false.

-- before vs after
select count(*) from ctable;
select count(*) from ch1;
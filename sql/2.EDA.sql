-- Built churn-focused EDA using CTEs and window functions, identifying high-risk segments based on service calls and international usage
-- Designed churn risk segmentation logic reducing ambiguity in customer intervention prioritization
-- Derived behavioral usage metrics and churn drivers to support retention-focused decision-making

-- overall churn rate
select * from ch1;

select count(*) as `total customer`,
sum(case when churn = 'True' then 1 else 0 end) as `churn customers`,
round(sum(case when churn = 'True' then 1 else 0 end)*100.0 / count(*),2) as `churn percentage`
from ch1;

-- international plan churned customer
select `International plan`,count(*) as `total customer`,
sum(case when churn = 'True' then 1 else 0 end) as `churn customers`,
round(sum(case when churn = 'True' then 1 else 0 end)*100.0 / count(*),2) as `churn percentage`
from ch1
group by `International plan`;

-- tenure of customer
WITH tenure_cte AS (
SELECT CASE 
	WHEN `account length` <= 12 THEN '0-1 years'
	WHEN `account length` BETWEEN 13 AND 24 THEN '1-2 years'
	WHEN `account length` BETWEEN 25 AND 36 THEN '2-3 years'
	ELSE '4+ years'
END AS tenure,churn
FROM ch1
),
tenure_metrics AS (
SELECT DISTINCT tenure,
	COUNT(*) OVER (PARTITION BY tenure) AS total_customers,
	SUM(churn = 'True') OVER (PARTITION BY tenure) AS churn_customers,
	ROUND(SUM(churn = 'True') OVER (PARTITION BY tenure) * 100.0 /COUNT(*) OVER (PARTITION BY tenure),2) AS churn_percentage
    FROM tenure_cte
)
SELECT
    tenure,
    total_customers,
    churn_customers,
    churn_percentage,
    LAG(churn_percentage) OVER (ORDER BY tenure) AS previous_churn_percentage
FROM tenure_metrics
ORDER BY tenure;

-- Average customer usage
select churn ,
round(avg(`Total day minutes`))  tot_day_min,
round(avg(`Total eve minutes`)) tot_eve_min,
round(avg(`Total night minutes`)) tot_night_min,
round(avg(`Total intl minutes`)) tot_intl_min
from ch1 group by Churn;

-- Usage-based revenue proxy assuming linear pricing model
select churn , 
round(avg(`Total day minutes`+`Total eve minutes`+`Total night minutes`+`Total intl minutes`),2) as total_usage_minutes
from ch1 group by churn;




-- Choosing threshold value 
SELECT `Customer service calls`, COUNT(*) AS customers
FROM ch1
GROUP BY `Customer service calls`
ORDER BY `Customer service calls`;

SELECT 
  `Customer service calls`,
  COUNT(*) AS total_customers,
  SUM(churn='True') AS churned_customers,
  ROUND(SUM(churn='True')*100.0/COUNT(*),2) AS churn_rate
FROM ch1
GROUP BY `Customer service calls`
ORDER BY `Customer service calls`;

-- customer service call 
select case 
	when `Customer service calls` <= 2 then 'Low'
    when `Customer service calls` between 3 and 5 then 'Medium'
    else 'High'
    end service_call
    ,count(*) as `total customer`,
sum( churn = 'True' ) as `churn customers`,
round(sum( churn = 'True' )*100.0 / count(*),2) as `churn percentage`
from ch1
group by service_call;

-- rank
with `rank` as
(SELECT 'Customer service calls >=4' AS factor,
ROUND(SUM(churn='True')*100.0/COUNT(*),2) AS churn_rate
FROM ch1
where `Customer service calls` >= 4

UNION ALL

SELECT 'International plan`=Yes' as factor,
ROUND(SUM(churn='True')*100.0/COUNT(*),2)
FROM ch1
where `International plan`='Yes')
select * from `rank` order by churn_rate desc;

-- Risk cte
WITH risk_cte AS (
SELECT CASE 
	WHEN `Customer service calls` >= 4 THEN 'High Risk – Immediate Intervention'
	WHEN `Customer service calls` BETWEEN 2 AND 3 THEN 'Medium Risk – Monitor'
	ELSE 'Low Risk – Stable'
    END AS risk_segment,
    COUNT(*) AS total_customers,
    SUM(churn='True') AS churned_customers,
    ROUND(SUM(churn='True')*100.0/COUNT(*),2) AS churn_rate
  FROM ch1
  GROUP BY risk_segment
)
SELECT * FROM risk_cte
ORDER BY churn_rate DESC;


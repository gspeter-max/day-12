''' 3220. Odd and Even Transactions ''' 

# Write your MySQL query statement
with temp_temp as (
    select transaction_date , 
    sum(case when amount  % 2 != 0 then amount else 0 end) as odd_sum,
    sum(case when amount % 2 = 0 then amount else 0 end ) as even_sum
    from transactions 
    group by transaction_date  
)
select transaction_date , 
    odd_sum,
    even_sum
from temp_temp
order by transaction_date; 


''' 3374. First Letter Capitalization II  (pandas ) '''
import pandas as pd

def capitalize_content(user_content: pd.DataFrame) -> pd.DataFrame:
    user_content['converted_texy'] = user_content['content_text'].apply(
        lambda x : x.title()
    )
    user_content.columns = ['content_id','original_text','converted_text']
    return user_content


''' 3421. Find Students Who Improved ''' 
with ranks as (
		select distinct student_id, subject,
				first_value(score) over(partition by  student_id, subject order by exam_date ) as first_score,
				first_value(score) over(partition by student_id, subject  order by exam_date desc ) as last_score

		from scores 
	) 
select student_id,
		subject, 
		first_score , 
		last_score as latest_score 
from ranks 
where first_score < last_score ; 

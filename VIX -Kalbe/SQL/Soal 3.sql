select s.storename, sum(t.qty) as total_qty from public.store s
join public."transaction" t on s.storeid = t.storeid 
group by s.storename 
order by total_qty desc
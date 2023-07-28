select p."Product Name" , sum(t.totalamount) as total_amount from public.product p
join public."transaction" t on p.productid = t.productid 
group by p."Product Name" 
order by total_amount desc
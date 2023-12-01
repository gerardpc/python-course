## TO DO

# Other operations

To kill a stuck process:

select pid, 
       usename, 
       pg_blocking_pids(pid) as blocked_by, 
       query as blocked_query
from pg_stat_activity
where cardinality(pg_blocking_pids(pid)) > 0;

select * from pg_catalog.pg_stat_activity where pid = 386423


SELECT pg_terminate_backend(<pid of the process>)

Optimitzar queries:

https://explain.dalibo.com/ i us explain(..) com explica a la web
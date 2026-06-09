create or replace table eciscor_prod.pcis_data_science.anm_training_data_matches as
with parties as (
  select
    identity_id,
    party_id,
    count(*) over (partition by identity_id) as cluster_size,
    row_number() over (partition by identity_id order by party_id) as rn
  from eciscor_prod.pcis_metadata.mv_es_party_identity
)
select
  a.identity_id as identityid,
  a.party_id as pr1,
  b.party_id as pr2,
  1 as label
from parties a
join parties b
  on a.identity_id = b.identity_id
 and a.rn < b.rn
where a.cluster_size between 2 and 10
  and rand(42) < 0.05;

-- =====================================================================

create or replace table eciscor_prod.pcis_data_science.anm_training_data_nonmatches as
with pool as (
  select party_id, identity_id
  from eciscor_prod.pcis_metadata.mv_es_party_identity
  tablesample (0.5 percent)
)
select
  a.identity_id as identityid,
  a.party_id as pr1,
  b.party_id as pr2,
  0 as label
from pool a
join pool b
  on a.party_id < b.party_id
 and a.identity_id <> b.identity_id
where rand(42) < 0.0005;

-- =====================================================================

create or replace table eciscor_prod.pcis_data_science.anm_training_data_nonmatches_hard as
with parties as (
  select
    i.party_id,
    i.identity_id,
    upper(substr(p.block_key, 1, 4)) as blk
  from eciscor_prod.pcis_metadata.mv_es_party_identity i
  join eciscor_prod.pcis_metadata.rpt_bronze_party p
    on p.party_id = i.party_id
)
select
  a.identity_id as identityid,
  a.party_id as pr1,
  b.party_id as pr2,
  0 as label
from parties a
join parties b
  on a.blk = b.blk
 and a.party_id < b.party_id
 and a.identity_id <> b.identity_id
where rand(42) < 0.01;

-- =====================================================================

create or replace table eciscor_prod.pcis_data_science.anm_training_data_fullset as
with pairs as (
  select identityid, pr1, pr2, label
  from eciscor_prod.pcis_data_science.anm_training_data_matches
  union all
  select identityid, pr1, pr2, label
  from eciscor_prod.pcis_data_science.anm_training_data_nonmatches
)
select
  pr.identityid,
  pr.pr1,
  pr.pr2,
  pr.label,
  b1.* except (party_id),
  b2.* except (party_id)
from pairs pr
join eciscor_prod.pcis_metadata.rpt_bronze_party b1 on b1.party_id = pr.pr1
join eciscor_prod.pcis_metadata.rpt_bronze_party b2 on b2.party_id = pr.pr2;

-- =====================================================================

create or replace table eciscor_prod.pcis_data_science.anm_training_data_sample as
select *
from eciscor_prod.pcis_data_science.anm_training_data_fullset
order by rand(42)
limit 8828;

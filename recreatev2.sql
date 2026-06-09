create or replace table eciscor_prod.pcis_data_science.anm_training_data_matches as
with parties as (
  select
    identityId,
    `_id` as party_id,
    row_number() over (partition by identityId order by `_id`) as rn,
    count(*) over (partition by identityId) as cluster_size
  from eciscor_prod.pcis_metadata.mv_es_party_identity
),
anchors as (
  select identityId, party_id as anchor_id
  from parties
  where rn = 1
)
select
  p.identityId as identityid,
  p.party_id as pr1,
  a.anchor_id as pr2,
  1 as label
from parties p
join anchors a on a.identityId = p.identityId
where p.cluster_size > 1
  and p.party_id <> a.anchor_id;

-- =====================================================================

create or replace table eciscor_prod.pcis_data_science.anm_training_data_nonmatches as
with anchors as (
  select identityId, min(`_id`) as anchor_id
  from eciscor_prod.pcis_metadata.mv_es_party_identity
  group by identityId
),
pool as (
  select identityId, anchor_id
  from anchors
  tablesample (2 percent)
)
select
  a.identityId as identityid,
  a.anchor_id as pr1,
  b.anchor_id as pr2,
  0 as label
from pool a
join pool b
  on a.anchor_id < b.anchor_id
 and a.identityId <> b.identityId
where rand(42) < 0.002;

-- =====================================================================

create or replace table eciscor_prod.pcis_data_science.anm_training_data_nonmatches_hard as
with parties as (
  select
    i.identityId,
    i.`_id` as party_id,
    upper(substr(get_json_object(p.`_source`, '$.lastName'), 1, 4)) as blk
  from eciscor_prod.pcis_metadata.mv_es_party_identity i
  join eciscor_prod.pcis_metadata.rpt_bronze_party p
    on p.`_id` = i.`_id`
)
select
  a.identityId as identityid,
  a.party_id as pr1,
  b.party_id as pr2,
  0 as label
from parties a
join parties b
  on a.blk = b.blk
 and a.party_id < b.party_id
 and a.identityId <> b.identityId
where rand(42) < 0.01;

-- =====================================================================

create or replace table eciscor_prod.pcis_data_science.anm_training_data_fullset as
with pairs as (
  select identityid, pr1, pr2, label
  from eciscor_prod.pcis_data_science.anm_training_data_matches
  union all
  select identityid, pr1, pr2, label
  from eciscor_prod.pcis_data_science.anm_training_data_nonmatches
),
src as (
  select `_id` as party_id, `_source` as doc
  from eciscor_prod.pcis_metadata.rpt_bronze_party
)
select
  p.identityid,
  p.pr1,
  p.pr2,
  p.label,
  get_json_object(s1.doc, '$.firstName') as pr1_first_name,
  get_json_object(s1.doc, '$.lastName')  as pr1_last_name,
  get_json_object(s1.doc, '$.dob')       as pr1_dob,
  get_json_object(s2.doc, '$.firstName') as pr2_first_name,
  get_json_object(s2.doc, '$.lastName')  as pr2_last_name,
  get_json_object(s2.doc, '$.dob')       as pr2_dob
from pairs p
join src s1 on s1.party_id = p.pr1
join src s2 on s2.party_id = p.pr2;

-- =====================================================================

create or replace table eciscor_prod.pcis_data_science.anm_training_data_sample as
select *
from eciscor_prod.pcis_data_science.anm_training_data_fullset
order by rand(42)
limit 8828;

/*+ SeqScan(ci)  Leading (ci) */ explain select count(*) from cast_info ci where  ci.person_id < 1700496;
/*+ IndexScan(ci)  Leading (ci) */ explain select count(*) from cast_info ci where  ci.person_id < 1700496;
/*+ BitmapScan(ci)  Leading (ci) */ explain select count(*) from cast_info ci where  ci.person_id < 1700496;
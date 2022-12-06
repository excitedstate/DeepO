/*+ SeqScan(ci)  Leading (ci) */ explain select count(*) from cast_info ci where  ci.person_id > 1423339;
/*+ IndexScan(ci)  Leading (ci) */ explain select count(*) from cast_info ci where  ci.person_id > 1423339;
/*+ BitmapScan(ci)  Leading (ci) */ explain select count(*) from cast_info ci where  ci.person_id > 1423339;
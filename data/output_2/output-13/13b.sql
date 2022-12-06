/*+ SeqScan(ci)  Leading (ci) */ explain select count(*) from cast_info ci where  ci.role_id = 10;
/*+ IndexScan(ci)  Leading (ci) */ explain select count(*) from cast_info ci where  ci.role_id = 10;
/*+ BitmapScan(ci)  Leading (ci) */ explain select count(*) from cast_info ci where  ci.role_id = 10;
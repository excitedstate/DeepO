/*+ SeqScan(ci)  Leading (ci) */ explain select count(*) from cast_info ci where  ci.role_id > 2;
/*+ IndexScan(ci)  Leading (ci) */ explain select count(*) from cast_info ci where  ci.role_id > 2;
/*+ BitmapScan(ci)  Leading (ci) */ explain select count(*) from cast_info ci where  ci.role_id > 2;
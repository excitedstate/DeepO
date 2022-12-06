/*+ SeqScan(t)  Leading (t) */ explain select count(*) from title t where  t.kind_id = 7 and t.production_year > 2011;
/*+ IndexScan(t)  Leading (t) */ explain select count(*) from title t where  t.kind_id = 7 and t.production_year > 2011;
/*+ BitmapScan(t)  Leading (t) */ explain select count(*) from title t where  t.kind_id = 7 and t.production_year > 2011;
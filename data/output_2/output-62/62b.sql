/*+ SeqScan(t)  Leading (t) */ explain select count(*) from title t where  t.production_year < 2012;
/*+ IndexScan(t)  Leading (t) */ explain select count(*) from title t where  t.production_year < 2012;
/*+ BitmapScan(t)  Leading (t) */ explain select count(*) from title t where  t.production_year < 2012;
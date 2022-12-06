/*+ SeqScan(t)  Leading (t) */ explain select count(*) from title t where  t.production_year = 1993;
/*+ IndexScan(t)  Leading (t) */ explain select count(*) from title t where  t.production_year = 1993;
/*+ BitmapScan(t)  Leading (t) */ explain select count(*) from title t where  t.production_year = 1993;
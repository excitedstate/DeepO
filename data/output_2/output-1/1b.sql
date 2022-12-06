/*+ SeqScan(t)  Leading (t) */ explain select count(*) from title t where  t.production_year > 2004;
/*+ IndexScan(t)  Leading (t) */ explain select count(*) from title t where  t.production_year > 2004;
/*+ BitmapScan(t)  Leading (t) */ explain select count(*) from title t where  t.production_year > 2004;
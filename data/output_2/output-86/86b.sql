/*+ SeqScan(t)  Leading (t) */ explain select count(*) from title t where  t.production_year > 2010;
/*+ IndexScan(t)  Leading (t) */ explain select count(*) from title t where  t.production_year > 2010;
/*+ BitmapScan(t)  Leading (t) */ explain select count(*) from title t where  t.production_year > 2010;
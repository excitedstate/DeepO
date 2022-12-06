/*+ SeqScan(t)  Leading (t) */ explain select count(*) from title t where  t.kind_id = 1;
/*+ IndexScan(t)  Leading (t) */ explain select count(*) from title t where  t.kind_id = 1;
/*+ BitmapScan(t)  Leading (t) */ explain select count(*) from title t where  t.kind_id = 1;
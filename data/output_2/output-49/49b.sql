/*+ SeqScan(t)  Leading (t) */ explain select count(*) from title t where  t.kind_id > 3;
/*+ IndexScan(t)  Leading (t) */ explain select count(*) from title t where  t.kind_id > 3;
/*+ BitmapScan(t)  Leading (t) */ explain select count(*) from title t where  t.kind_id > 3;
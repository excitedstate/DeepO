/*+ SeqScan(mc)  Leading (mc) */ explain select count(*) from movie_companies mc where  mc.company_type_id < 2;
/*+ IndexScan(mc)  Leading (mc) */ explain select count(*) from movie_companies mc where  mc.company_type_id < 2;
/*+ BitmapScan(mc)  Leading (mc) */ explain select count(*) from movie_companies mc where  mc.company_type_id < 2;
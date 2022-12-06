/*+ SeqScan(mc)  Leading (mc) */ explain select count(*) from movie_companies mc where  mc.company_id < 95839;
/*+ IndexScan(mc)  Leading (mc) */ explain select count(*) from movie_companies mc where  mc.company_id < 95839;
/*+ BitmapScan(mc)  Leading (mc) */ explain select count(*) from movie_companies mc where  mc.company_id < 95839;
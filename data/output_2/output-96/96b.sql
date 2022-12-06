/*+ SeqScan(mi)  Leading (mi) */ explain select count(*) from movie_info mi where  mi.info_type_id = 5;
/*+ IndexScan(mi)  Leading (mi) */ explain select count(*) from movie_info mi where  mi.info_type_id = 5;
/*+ BitmapScan(mi)  Leading (mi) */ explain select count(*) from movie_info mi where  mi.info_type_id = 5;
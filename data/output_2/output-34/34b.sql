/*+ SeqScan(mi_idx)  Leading (mi_idx) */ explain select count(*) from movie_info_idx mi_idx where  mi_idx.info_type_id = 101;
/*+ IndexScan(mi_idx)  Leading (mi_idx) */ explain select count(*) from movie_info_idx mi_idx where  mi_idx.info_type_id = 101;
/*+ BitmapScan(mi_idx)  Leading (mi_idx) */ explain select count(*) from movie_info_idx mi_idx where  mi_idx.info_type_id = 101;
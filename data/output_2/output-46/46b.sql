/*+ SeqScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id < 3639;
/*+ IndexScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id < 3639;
/*+ BitmapScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id < 3639;
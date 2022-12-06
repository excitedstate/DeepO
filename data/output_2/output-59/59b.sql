/*+ SeqScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id > 2758;
/*+ IndexScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id > 2758;
/*+ BitmapScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id > 2758;
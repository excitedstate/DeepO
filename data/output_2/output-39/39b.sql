/*+ SeqScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id > 5071;
/*+ IndexScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id > 5071;
/*+ BitmapScan(mk)  Leading (mk) */ explain select count(*) from movie_keyword mk where  mk.keyword_id > 5071;
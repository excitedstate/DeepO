explain select count(*) from title t, movie_info mi, movie_keyword mk where t.id=mi.movie_id and t.id=mk.movie_id and t.kind_id = 7 and t.production_year < 1983 and mi.info_type_id = 18;
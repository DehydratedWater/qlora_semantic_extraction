
CREATE Table IF NOT EXISTS articles (
    article_id INT NOT NULL,
    dataset TEXT NOT NULL,
    abstract TEXT NOT NULL,
    section_names TEXT NOT NULL,
    PRIMARY KEY (article_id)
);

CREATE TABLE if not exists article_part_register (
    part_index INT NOT NULL,
    chunk_size INT NOT NULL,
    chunk_overlap INT NOT NULL,
    article_id INT NOT NULL,
    part_id SERIAL NOT NULL,
    UNIQUE (part_index, chunk_size, chunk_overlap, article_id),
    PRIMARY KEY (part_id),
    CONSTRAINT fk_apr_article_id
        FOREIGN KEY(article_id) 
            REFERENCES articles(article_id)
);

CREATE TABLE if not exists article_parts (
    part_id INT NOT NULL,
    part_text TEXT NOT NULL,
    PRIMARY KEY (part_id),
    CONSTRAINT fk_ap_part_id
        FOREIGN KEY(part_id) 
            REFERENCES article_part_register(part_id)
);

CREATE TABLE if not exists short_article_summary (
    article_id INT NOT NULL,
    article_summary TEXT NOT NULL,
    summary_variant INT NOT NULL,
    summary_id SERIAL NOT NULL,
    PRIMARY KEY (article_id, summary_variant),
    UNIQUE (summary_id),
    CONSTRAINT fk_sas_article_id
        FOREIGN KEY(article_id) 
            REFERENCES articles(article_id)
);

CREATE TABLE if not exists short_part_summary (
    part_id INT NOT NULL,
    part_summary TEXT NOT NULL,
    summary_variant INT NOT NULL,
    summary_id SERIAL NOT NULL,
    article_summary_id INT NOT NULL,
    PRIMARY KEY (part_id, summary_variant),
    UNIQUE (summary_id),
    CONSTRAINT fk_sps_part_id
        FOREIGN KEY(part_id) 
            REFERENCES article_parts(part_id),
    CONSTRAINT fk_sps_article_summary_id
        FOREIGN KEY(article_summary_id) 
            REFERENCES short_article_summary(summary_id)
);

CREATE TABLE if not exists extracted_part_topics (
    part_id INT NOT NULL,
    part_topics TEXT NOT NULL,
    topics_variant INT NOT NULL,
    topics_id SERIAL NOT NULL,
    article_summary_id INT NOT NULL,
    PRIMARY KEY (part_id, topics_variant),
    UNIQUE (topics_id),
    CONSTRAINT fk_ept_part_id
        FOREIGN KEY(part_id) 
            REFERENCES article_parts(part_id),
    CONSTRAINT fk_ept_article_summary_id
        FOREIGN KEY(article_summary_id) 
            REFERENCES short_article_summary(summary_id)
);

CREATE TABLE if not exists extracted_relations_raw (
    article_id INT NOT NULL,
    part_id INT NOT NULL,
    part_summary_id INT NOT NULL,
    part_topics_id INT NOT NULL,
    article_summary_id INT NOT NULL,
    relation_id SERIAL NOT NULL,
    raw_relation_text TEXT NOT NULL,
    relations_variant INT NOT NULL,
    PRIMARY KEY (relation_id, relations_variant),
    UNIQUE (relation_id),
    CONSTRAINT fk_err_article_id
        FOREIGN KEY(article_id) 
            REFERENCES articles(article_id),
    CONSTRAINT fk_err_part_id
        FOREIGN KEY(part_id) 
            REFERENCES article_parts(part_id),
    CONSTRAINT fk_err_part_summary_id
        FOREIGN KEY(part_summary_id) 
            REFERENCES short_part_summary(summary_id),
    CONSTRAINT fk_err_part_topics_id
        FOREIGN KEY(part_topics_id) 
            REFERENCES extracted_part_topics(topics_id),
    CONSTRAINT fk_err_article_summary_id
        FOREIGN KEY(article_summary_id) 
            REFERENCES short_article_summary(summary_id)
);

CREATE TABLE if not exists relation_objects (
    source_id INT NOT NULL,
    object_id SERIAL NOT NULL,
    object_name TEXT NOT NULL,
    PRIMARY KEY (object_id, object_name),
    UNIQUE (object_id),
    CONSTRAINT fk_ro_relation_id
        FOREIGN KEY(source_id) 
            REFERENCES extracted_relations_raw(relation_id)
);

CREATE TABLE if not exists relations (
    relation_id SERIAL NOT NULL,
    source_id INT NOT NULL,
    object_id INT NOT NULL,
    subject_id INT NOT NULL,
    relation_type TEXT NOT NULL,
    PRIMARY KEY (relation_id),
    CONSTRAINT fk_r_relation_id
        FOREIGN KEY(source_id) 
            REFERENCES extracted_relations_raw(relation_id),
    CONSTRAINT fk_r_object_id
        FOREIGN KEY(object_id) 
            REFERENCES relation_objects(object_id),
    CONSTRAINT fk_r_subject_id
        FOREIGN KEY(subject_id) 
            REFERENCES relation_objects(object_id)
    
);
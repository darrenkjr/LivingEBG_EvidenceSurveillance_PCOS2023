-- 1st Level Tables 

CREATE TABLE gdgs (
    gdg_id INT PRIMARY KEY, 
    topic TEXT
    );

CREATE TABLE databases (
    database_id INT PRIMARY KEY, 
    database_name TEXT, 
    free_api_available BOOLEAN
    );

CREATE TABLE search_types (
    search_type_id INT PRIMARY KEY, 
    name TEXT
    );

CREATE TABLE search_strategy_types (
    search_strategy_type_id INT PRIMARY KEY, 
    name TEXT
    );

-- 2nd Level Tables 

CREATE TABLE evidence_reviews (
    evidence_review_id VARCHAR(50) PRIMARY KEY, 
    gdg_id INT REFERENCES gdgs(gdg_id), 
    question TEXT, 
    evidence_review_type TEXT, 
    included_num INT
    );

-- 3rd level tables 
CREATE TABLE ground_truth_articles (
    ground_truth_article_id INT PRIMARY KEY, 
    evidence_review_id VARCHAR(50) REFERENCES evidence_reviews(evidence_review_id), 
    included_reference TEXT, 
    author_year_format VARCHAR(200),
    extracted_publication_year INT, 
    assessed_rob VARCHAR(25), 
    retrieved_oa_id VARCHAR(50), 
    retrieved_embase_id VARCHAR(50), 
    retrieved_pubmed_id VARCHAR(50),
    title TEXT,
    abstract TEXT 
    );

CREATE TABLE search_strategies (
    search_strategy_id INT PRIMARY KEY, 
    evidence_review_id VARCHAR(50) REFERENCES evidence_reviews(evidence_review_id), 
    database_id INT REFERENCES databases(database_id), 
    searchstrat_year_start INT, 
    searchstrat_year_end INT, 
    searchdetail_file_path TEXT, 
    search_type_id INT REFERENCES search_types(search_type_id), 
    search_strategy_type_id INT REFERENCES search_strategy_types(search_strategy_type_id)
    );

-- 4th level tables 

CREATE TABLE evaluation_results (
    evaluation_result_id INT PRIMARY KEY, 
    search_strategy_id INT REFERENCES search_strategies(search_strategy_id), 
    recall FLOAT, 
    precision FLOAT, 
    f1_score FLOAT,
    f2_score FLOAT,
    f3_score FLOAT
    );

CREATE TABLE search_result_articles (
    search_result_article_id INT PRIMARY KEY, 
    search_strategy_id INT REFERENCES search_strategies(search_strategy_id);
    title TEXT, 
    abstract TEXT, 
    publication_year INT,
    original_id VARCHAR(50) 
    );

-- foreign key constraints with RESTRICT
ALTER TABLE evidence_reviews 
ADD CONSTRAINT fk_gdg 
FOREIGN KEY (gdg_id) 
REFERENCES gdgs(gdg_id)
ON DELETE RESTRICT  -- Prevent deletion of GDG if it has evidence reviews
ON UPDATE RESTRICT;

ALTER TABLE search_strategies 
ADD CONSTRAINT fk_evidence_review 
FOREIGN KEY (evidence_review_id) 
REFERENCES evidence_reviews(evidence_review_id)
ON DELETE RESTRICT  -- Prevent deletion of evidence review if it has strategies
ON UPDATE RESTRICT;


ALTER TABLE search_result_articles 
ADD CONSTRAINT fk_search_strategy 
FOREIGN KEY (search_strategy_id) 
REFERENCES search_strategies(search_strategy_id)
ON DELETE RESTRICT
ON UPDATE RESTRICT;
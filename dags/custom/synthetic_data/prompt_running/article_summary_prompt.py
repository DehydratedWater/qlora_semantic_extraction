   
article_summary_prompt = """
[INST] 
Provided below abstract and list of sections of the article, generate summary for provided text. Summary should contain not only most important sentences, but also most important ideas, concepts, people, ect.

To generate summary use provided abstract: ```
{abstract}
```

Use also provided list of sections: ```
{list_of_sections}
```

Output your summary, without any additional text, in the following format Summary: ```summary <<summary text goes gere>>```

[/INST]
Summary:
"""
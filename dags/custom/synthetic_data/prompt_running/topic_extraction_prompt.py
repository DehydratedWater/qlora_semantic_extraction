   
extracton_prompt = """
[INST] 
Provided below segment of the text, list all specialistic terms, and ideas contained in this text, that list will be later used for mining relations between concepts in order of building graph of semantic relations. 

Example:
Format data as a JSON containing {{"segment_summary": "summary ...", extracted_categories:{{"<<name of category>>": [{{"description": "What is idea number 1...","name": "idea_1"}}, {{"description": "What is term number 2...","name": "term_2"}}, ...], "<<name of category>>": [{{"description": "What is idea number 1...","name": "idea_1"}}, {{"description": "What is term number 2...","name": "term_2"}}, {{"description": "What is idea number 3...","name": "idea_3"}}...]}}. Categories should be general.

Text to extract categories: ```
{text_to_extract}
```

Use text above to extract concepts, ideas, people, ect, find name for the category and format them into flat JSON containing lists. Return JSON within 
json``` {{...}} ```
[/INST]
Before creating JSON with the short segment summary and results, containing categories matched with context independent description of interaction, I will list categories that fits into nodes of semantic graph, contained inside provided text and give them more compact and universal name:
"""
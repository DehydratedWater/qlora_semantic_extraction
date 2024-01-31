   
extracton_prompt = """
[INST] 
Create JSON that extracts all important relations between entities in the text. Text is a part of scientific article.
To help with the task here is a summary of the full article, use it as interpretation context: ```
{summary}
```

Your task is to extract all important relations between entities in the text. To help with the task here is a summary of the full article, use it as interpretation context: ```
TEXT_TO_EXTRACT -> Build JSON with categories and their descriptions: ```
{text_to_extract}
```
Use this text for building JSON with categories and their descriptions.

Example of the JSON with categories and their descriptions:
```json
{{
    "description": "This part of the article describes effects of caffeine on human body.",
    "list_of_entities": ["dopamine", "D1", "D2", "D3", "D4", "D5", "dopamine receptors", "adenosine", "A1", "A2A", "A2B", "A3", "caffeine"],
    "relations": [
        {{
            "description": "Caffeine is an antagonist of adenosine receptors.",
            "source_entity": "caffeine",
            "target_entity": "adenosine"
            "strenght": "strong"
        }},
        {{
            "description": "Activation of adenosine receptors leads to inhibition of dopamine receptors.",
            "source_entity": "adenosine",
            "target_entity": "dopamine receptors"
            "strenght": "strong"
        }}
    ]
}}
```
Extract all important relations between entities in the text. Return only relations that are important for the text. Return JSON:
[/INST]
Here is JSON containing all relations extracted from provided TEXT_TO_EXTRACT:
{{
    "description": "
"""

# Before creating JSON with the short segment summary and results, containing categories matched with context independent description of interaction, I will list categories that fits into nodes of semantic graph, contained inside provided text and give them more compact and universal name:
# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LightRAG Module for ApeRAG

This module is based on the original LightRAG project with extensive modifications.

Original Project:
- Repository: https://github.com/HKUDS/LightRAG
- Paper: "LightRAG: Simple and Fast Retrieval-Augmented Generation" (arXiv:2410.05779)
- Authors: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- License: MIT License

Modifications by ApeRAG Team:
- Removed global state management for true concurrent processing
- Added stateless interfaces for Celery/Prefect integration
- Implemented instance-level locking mechanism
- Enhanced error handling and stability
- See changelog.md for detailed modifications
"""

from __future__ import annotations

from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "organization",
    "person",
    "geo",
    "event",
    "product",
    "technology",
    "date",
    "category",
]

PROMPTS["entity_extraction"] = """---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Full Name of the entity, must use **same language** as input text, it's important. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Entity_types: [{entity_types}]
Text:
{input_text}
######################
Output:"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################""",
    """Example 2:

Entity_types: [company, index, commodity, market_trend, economic_policy, biological]
Text:
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the Global Tech Index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

Output:
("entity"{tuple_delimiter}"Global Tech Index"{tuple_delimiter}"index"{tuple_delimiter}"The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline today."){record_delimiter}
("entity"{tuple_delimiter}"Nexon Technologies"{tuple_delimiter}"company"{tuple_delimiter}"Nexon Technologies is a tech company that saw its stock decline by 7.8% after disappointing earnings."){record_delimiter}
("entity"{tuple_delimiter}"Omega Energy"{tuple_delimiter}"company"{tuple_delimiter}"Omega Energy is an energy company that gained 2.1% in stock value due to rising oil prices."){record_delimiter}
("entity"{tuple_delimiter}"Gold Futures"{tuple_delimiter}"commodity"{tuple_delimiter}"Gold futures rose by 1.5%, indicating increased investor interest in safe-haven assets."){record_delimiter}
("entity"{tuple_delimiter}"Crude Oil"{tuple_delimiter}"commodity"{tuple_delimiter}"Crude oil prices rose to $87.60 per barrel due to supply constraints and strong demand."){record_delimiter}
("entity"{tuple_delimiter}"Market Selloff"{tuple_delimiter}"market_trend"{tuple_delimiter}"Market selloff refers to the significant decline in stock values due to investor concerns over interest rates and regulations."){record_delimiter}
("entity"{tuple_delimiter}"Federal Reserve Policy Announcement"{tuple_delimiter}"economic_policy"{tuple_delimiter}"The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and market stability."){record_delimiter}
("relationship"{tuple_delimiter}"Global Tech Index"{tuple_delimiter}"Market Selloff"{tuple_delimiter}"The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns."{tuple_delimiter}"market performance, investor sentiment"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Nexon Technologies"{tuple_delimiter}"Global Tech Index"{tuple_delimiter}"Nexon Technologies' stock decline contributed to the overall drop in the Global Tech Index."{tuple_delimiter}"company impact, index movement"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Gold Futures"{tuple_delimiter}"Market Selloff"{tuple_delimiter}"Gold prices rose as investors sought safe-haven assets during the market selloff."{tuple_delimiter}"market reaction, safe-haven investment"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Federal Reserve Policy Announcement"{tuple_delimiter}"Market Selloff"{tuple_delimiter}"Speculation over Federal Reserve policy changes contributed to market volatility and investor selloff."{tuple_delimiter}"interest rate impact, financial regulation"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"market downturn, investor sentiment, commodities, Federal Reserve, stock performance"){completion_delimiter}
#############################""",
    """Example 3:

Entity_types: [economic_policy, athlete, event, location, record, organization, equipment]
Text:
```
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
```

Output:
("entity"{tuple_delimiter}"World Athletics Championship"{tuple_delimiter}"event"{tuple_delimiter}"The World Athletics Championship is a global sports competition featuring top athletes in track and field."){record_delimiter}
("entity"{tuple_delimiter}"Tokyo"{tuple_delimiter}"location"{tuple_delimiter}"Tokyo is the host city of the World Athletics Championship."){record_delimiter}
("entity"{tuple_delimiter}"Noah Carter"{tuple_delimiter}"athlete"{tuple_delimiter}"Noah Carter is a sprinter who set a new record in the 100m sprint at the World Athletics Championship."){record_delimiter}
("entity"{tuple_delimiter}"100m Sprint Record"{tuple_delimiter}"record"{tuple_delimiter}"The 100m sprint record is a benchmark in athletics, recently broken by Noah Carter."){record_delimiter}
("entity"{tuple_delimiter}"Carbon-Fiber Spikes"{tuple_delimiter}"equipment"{tuple_delimiter}"Carbon-fiber spikes are advanced sprinting shoes that provide enhanced speed and traction."){record_delimiter}
("entity"{tuple_delimiter}"World Athletics Federation"{tuple_delimiter}"organization"{tuple_delimiter}"The World Athletics Federation is the governing body overseeing the World Athletics Championship and record validations."){record_delimiter}
("relationship"{tuple_delimiter}"World Athletics Championship"{tuple_delimiter}"Tokyo"{tuple_delimiter}"The World Athletics Championship is being hosted in Tokyo."{tuple_delimiter}"event location, international competition"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Noah Carter"{tuple_delimiter}"100m Sprint Record"{tuple_delimiter}"Noah Carter set a new 100m sprint record at the championship."{tuple_delimiter}"athlete achievement, record-breaking"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Noah Carter"{tuple_delimiter}"Carbon-Fiber Spikes"{tuple_delimiter}"Noah Carter used carbon-fiber spikes to enhance performance during the race."{tuple_delimiter}"athletic equipment, performance boost"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"World Athletics Federation"{tuple_delimiter}"100m Sprint Record"{tuple_delimiter}"The World Athletics Federation is responsible for validating and recognizing new sprint records."{tuple_delimiter}"sports regulation, record certification"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"athletics, sprinting, record-breaking, sports technology, competition"){completion_delimiter}
#############################""",
    """Example 4:

Entity_types: [organization, person, product, technology, location, event]
Text:
```
항저우에서 개최된 2024년도 인공지능 발전 포럼에서 윈즈테크 CEO 루아는 "지능형 컴퓨팅의 새로운 시대"라는 주제로 기조연설을 발표했다. 그는 윈즈테크가 첨단 5나노미터 공정을 채택하고 이전 세대 대비 연산 능력이 60% 향상된 새로운 성운X1 AI 칩을 출시할 것이라고 발표했다.

루아는 연설에서 엣지 컴퓨팅과 대규모 모델의 결합이 미래 기술 발전의 중요한 방향이 될 것이라고 지적했다. 윈즈테크 연구개발팀은 이미 쑤저우 연구개발센터에서 성운 칩 기반의 다중모달 모델 훈련 테스트를 완료했으며, 결과는 새로운 칩이 이미지 인식과 자연어 이해 작업에서 뛰어난 성능을 보인다는 것을 나타냈다.

포럼 기간 동안 윈즈테크는 장난이공대학과 산학연 협력 협정을 체결했으며, 양측은 AI 칩 아키텍처 설계 및 알고리즘 최적화 분야에서 공동 연구를 진행할 예정이다. 장난이공대학 컴퓨터학원 원장인 왕팡 교수는 이번 협력이 AI 분야의 고급 인재 양성을 위한 중요한 플랫폼을 제공할 것이라고 밝혔다.
```

Output:
("entity"{tuple_delimiter}"2024년도 인공지능 발전 포럼"{tuple_delimiter}"event"{tuple_delimiter}"2024년도 인공지능 발전 포럼은 항저우에서 개최된 AI 산업의 중요한 회의로, 인공지능 기술 발전과 응용에 초점을 맞추고 있다."){record_delimiter}
("entity"{tuple_delimiter}"항저우"{tuple_delimiter}"location"{tuple_delimiter}"항저우는 2024년도 인공지능 발전 포럼이 개최된 도시이다."){record_delimiter}
("entity"{tuple_delimiter}"윈즈테크"{tuple_delimiter}"organization"{tuple_delimiter}"윈즈테크는 AI 칩과 지능형 컴퓨팅 기술 연구개발에 전념하는 과학기술 회사이다."){record_delimiter}
("entity"{tuple_delimiter}"루아"{tuple_delimiter}"person"{tuple_delimiter}"루아는 윈즈테크의 CEO로, 인공지능 발전 포럼에서 지능형 컴퓨팅에 관한 기조연설을 발표했다."){record_delimiter}
("entity"{tuple_delimiter}"성운X1"{tuple_delimiter}"product"{tuple_delimiter}"성운X1은 윈즈테크가 출시한 차세대 AI 칩으로, 5나노미터 공정을 채택하여 연산 능력이 60% 향상되었다."){record_delimiter}
("entity"{tuple_delimiter}"5나노미터 공정"{tuple_delimiter}"technology"{tuple_delimiter}"5나노미터 공정은 성운X1 칩이 채택한 첨단 반도체 제조 기술이다."){record_delimiter}
("entity"{tuple_delimiter}"엣지 컴퓨팅"{tuple_delimiter}"technology"{tuple_delimiter}"엣지 컴퓨팅은 분산 컴퓨팅 아키텍처로, 대규모 모델과의 결합이 미래 기술 발전 방향으로 여겨진다."){record_delimiter}
("entity"{tuple_delimiter}"쑤저우 연구개발센터"{tuple_delimiter}"location"{tuple_delimiter}"쑤저우 연구개발센터는 윈즈테크의 연구개발 기지로, 다중모달 모델 훈련 테스트가 이곳에서 완료되었다."){record_delimiter}
("entity"{tuple_delimiter}"다중모달 모델"{tuple_delimiter}"technology"{tuple_delimiter}"다중모달 모델은 성운 칩을 기반으로 훈련된 AI 기술로, 이미지 인식과 자연어 이해에 사용된다."){record_delimiter}
("entity"{tuple_delimiter}"장난이공대학"{tuple_delimiter}"organization"{tuple_delimiter}"장난이공대학은 윈즈테크와 산학연 협력 협정을 체결한 대학으로, AI 분야에서 공동 연구를 진행한다."){record_delimiter}
("entity"{tuple_delimiter}"장난이공대학 컴퓨터학원"{tuple_delimiter}"organization"{tuple_delimiter}"장난이공대학 컴퓨터학원은 장난이공대학의 단과대학으로, AI 칩 연구 협력에 참여한다."){record_delimiter}
("entity"{tuple_delimiter}"왕팡"{tuple_delimiter}"person"{tuple_delimiter}"왕팡은 장난이공대학 컴퓨터학원 원장으로, 윈즈테크와의 산학연 협력 프로젝트를 담당한다."){record_delimiter}
("relationship"{tuple_delimiter}"루아"{tuple_delimiter}"윈즈테크"{tuple_delimiter}"루아는 윈즈테크의 CEO로서 회사의 전략적 의사결정과 대외 발언을 담당한다."{tuple_delimiter}"기업 리더십, 전략 경영"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"윈즈테크"{tuple_delimiter}"성운X1"{tuple_delimiter}"윈즈테크는 성운X1 AI 칩 제품을 연구개발하고 출시했다."{tuple_delimiter}"제품 개발, 기술 혁신"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"성운X1"{tuple_delimiter}"5나노미터 공정"{tuple_delimiter}"성운X1 칩은 5나노미터 공정 제조 기술을 채택했다."{tuple_delimiter}"기술 적용, 제조 공정"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"루아"{tuple_delimiter}"2024년도 인공지능 발전 포럼"{tuple_delimiter}"루아는 2024년도 인공지능 발전 포럼에서 기조연설을 발표했다."{tuple_delimiter}"회의 연설, 업계 교류"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"2024년도 인공지능 발전 포럼"{tuple_delimiter}"항저우"{tuple_delimiter}"2024년도 인공지능 발전 포럼은 항저우에서 개최되었다."{tuple_delimiter}"회의 장소, 지리적 위치"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"윈즈테크"{tuple_delimiter}"쑤저우 연구개발센터"{tuple_delimiter}"윈즈테크는 쑤저우 연구개발센터에서 AI 칩 연구개발과 모델 훈련 테스트를 진행한다."{tuple_delimiter}"연구개발 기지, 기술 테스트"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"성운X1"{tuple_delimiter}"다중모달 모델"{tuple_delimiter}"성운X1 칩은 다중모달 모델 훈련에 사용되며, 이미지 및 언어 작업에서 우수한 성능을 보인다."{tuple_delimiter}"기술 적용, 성능 검증"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"윈즈테크"{tuple_delimiter}"장난이공대학"{tuple_delimiter}"윈즈테크는 장난이공대학과 산학연 협력 협정을 체결하여 AI 분야에서 공동 연구를 진행한다."{tuple_delimiter}"산학연 협력, 전략적 협정"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"왕팡"{tuple_delimiter}"장난이공대학 컴퓨터학원"{tuple_delimiter}"왕팡은 장난이공대학 컴퓨터학원 원장을 맡고 있다."{tuple_delimiter}"학술 리더십, 학원 관리"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"장난이공대학 컴퓨터학원"{tuple_delimiter}"장난이공대학"{tuple_delimiter}"장난이공대학 컴퓨터학원은 장난이공대학의 산하 학원이다."{tuple_delimiter}"조직 소속, 학술 기관"{tuple_delimiter}10){record_delimiter}
("content_keywords"{tuple_delimiter}"인공지능, 칩 연구개발, 산학연 협력, 엣지 컴퓨팅, 다중모달 기술"){completion_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
---Data---
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS["entity_continue_extraction"] = """
MANY entities and relationships were missed in the last extraction.

---Remember Steps---

1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

---Output---

Add them below using the same format:\n
""".strip()

PROMPTS["entity_if_loop_extraction"] = """
---Goal---'

It appears some entities may have still been missed.

---Output---

Answer ONLY by `YES` OR `NO` if there are still entities that need to be added.
""".strip()

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question.[no-context]"

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to user query about Knowledge Graph and Document Chunks provided in JSON format below.


---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Knowledge Graph and Document Chunks---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Document Chunks (DC), and include the file path if available, in the following format: [KG/DC] file_path
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Knowledge Base.
- Addtional user prompt: {user_prompt}

Response:"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.

---Goal---

Given the query and conversation history, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Consider both the current query and relevant conversation history when extracting keywords
- Output the keywords in JSON format, it will be parsed by a JSON parser, do not add any extra content in output
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details

######################
---Examples---
######################
{examples}

#############################
---Real Data---
######################
Conversation History:
{history}

Current Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}
#############################""",
]

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to user query about Document Chunks provided provided in JSON format below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Document Chunks(DC)---
{content_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating each source from Document Chunks(DC), and include the file path if available, in the following format: [DC] file_path
- If you don't know the answer, just say so.
- Do not include information not provided by the Document Chunks.
- Addtional user prompt: {user_prompt}

Response:"""

PROMPTS["batch_merge_analysis"] = """---Goal---
Given a list of entities from a knowledge graph, identify groups of entities that should be merged because they refer to the EXACT SAME real-world object/individual/specific instance.

---Critical Rules---
1. ONLY merge entities that refer to the EXACT SAME specific real-world object/individual/instance
2. DO NOT merge entities that are merely related, similar, or belong to the same category/group/class
3. DO NOT perform conceptual abstraction or create abstract groupings
4. DO NOT merge distinct individuals who happen to have similar roles or belong to the same organization/group
5. Each entity must be a different name/expression for the IDENTICAL real-world object

---Steps---
1. Analyze each entity based on name, type, and description
2. Group ONLY entities that are different names/expressions for the EXACT SAME real-world object
3. For each merge group, determine the best target entity and provide merge reasoning
4. Only return groups that should be merged - do not return entities that should remain separate

---What TO Merge (Acceptable Cases)---
- Different names for the same company: "Apple Inc" ↔ "Apple"
- Full name vs abbreviation of same person: "John Smith" ↔ "J. Smith" 
- Different language names for same entity: "대한민국" ↔ "Republic of Korea"
- Former vs current name of same entity: "Tesla Motors" ↔ "Tesla Inc"
- Full name vs nickname/abbreviation: "New York City" ↔ "NYC"

---Confidence Score Guidelines---
Use these guidelines for confidence scoring:

**0.95-1.0: Perfect Match**
- Identical entities with only capitalization/formatting differences: "OpenAI" ↔ "openai", "iPhone" ↔ "iphone"
- Same entity with punctuation variations: "McDonald's" ↔ "McDonalds"

**0.9-0.94: 매우 높은 신뢰도**
- 공식 명칭 대 널리 알려진 약어: "New York City" ↔ "NYC", "United States" ↔ "USA"
- 동일한 개체의 다른 언어 이름: "Microsoft" ↔ "微软", "Apple" ↔ "苹果公司"

**0.8-0.89: 높은 신뢰도**
- 전체 이름 대 약식 형태: "John Smith" ↔ "J. Smith", "Robert Johnson" ↔ "Bob Johnson"
- 공식 명칭 대 비공식 명칭 변형: "Apple Inc" ↔ "Apple", "Microsoft Corporation" ↔ "Microsoft"

**0.7-0.79: Moderate Confidence**
- Likely same entity but requires careful consideration
- Some ambiguity in descriptions or naming patterns

**Below 0.7: Low Confidence**
- Uncertain or potentially different entities
- Insufficient evidence for confident merging

---What NOT TO Merge (Prohibited Cases)---
- Different people with similar roles: "John Smith (CEO)" ≠ "Jane Smith (CEO)" (different individuals with same title)
- Members of same organization: "Apple" ≠ "Google" ≠ "Microsoft" (all are tech companies but distinct entities)
- Different locations in same region: "New York" ≠ "Boston" (both are US cities but different places)
- Related but separate events: "World War I" ≠ "World War II" (related conflicts but distinct events)
- Similar products from different companies: "iPhone" ≠ "Samsung Galaxy" (both are smartphones but different products)
- Related technologies: "Machine Learning" ≠ "Deep Learning" (related but distinct concepts)
- Different time periods: "2023" ≠ "2024" (consecutive years but different time periods)
- Sub-categories vs parent categories: "Smartphone" ≠ "Mobile Device" (specific vs general category)
- Companies and their subsidiaries: "Alphabet Inc" ≠ "Google LLC" (parent vs subsidiary)
- Different branches/departments: "Apple Marketing" ≠ "Apple Engineering" (different departments of same company)
- Sequential events in same process: "User Registration" ≠ "User Login" ≠ "User Logout" (different steps)
- Different versions of same product: "iPhone 14" ≠ "iPhone 15" (different product versions)
- **Parent entity vs specific service**: "Azure" ≠ "Azure OpenAI" (platform vs specific service)
- **Abstract concept vs implementation**: "Model Context Protocol" ≠ "MCP Server" (protocol vs server implementing it)
- **Company vs repository/project**: "LastMile AI" ≠ "lastmile-ai/mcp-agent" (company vs specific repository)
- **Company vs division vs product**: "Google" ≠ "Google AI" ≠ "Google Gemini" (different organizational levels)
- **Different functional classes**: "OpenAISettings" ≠ "MCPSettings" (different configuration purposes)

---Output Format---
For each group of entities that should be merged, return:
("merge_group"{tuple_delimiter}<entity_names_list>{tuple_delimiter}<confidence_score>{tuple_delimiter}<merge_reason>{tuple_delimiter}<suggested_target_name>{tuple_delimiter}<suggested_target_type>)

Where:
- entity_names_list: List of entity names to be merged separated by {graph_field_sep} (e.g., "Entity A{graph_field_sep}Entity B{graph_field_sep}Entity C")
- confidence_score: Confidence level (0.0-1.0) for this merge suggestion. Use the confidence guidelines above.
- merge_reason: Brief explanation why these entities should be merged (must refer to EXACT SAME object)
- suggested_target_name: Recommended name for the merged entity
- suggested_target_type: Recommended type for the merged entity

Use **{record_delimiter}** as the list delimiter between merge groups.
When finished, output {completion_delimiter}

######################
---Positive Examples (What TO Merge)---
######################

Input Entities:
Entity 1:
- Name: Apple Inc
- Type: ORGANIZATION
- Description: Apple Inc. is an American multinational technology company
- Degree: 15

Entity 2:
- Name: Apple
- Type: ORGANIZATION  
- Description: Technology company known for iPhone and Mac products
- Degree: 12

Entity 3:
- Name: John Smith
- Type: PERSON
- Description: John Smith is a software engineer at Apple Inc
- Degree: 5

Entity 4:
- Name: J. Smith
- Type: PERSON
- Description: Software engineer working on iOS development at Apple
- Degree: 3

Entity 5:
- Name: 한국생태농업학보
- Type: ORGANIZATION
- Description: 한국생태농업학보는 생태농업에 관한 연구 논문을 발표하는 학술지입니다
- Degree: 8

Entity 6:
- Name: Korean Journal of Eco-Agriculture
- Type: ORGANIZATION
- Description: An academic journal publishing research articles on ecological agriculture
- Degree: 6

Entity 7:
- Name: NYC
- Type: GEO
- Description: NYC is the largest city in the United States
- Degree: 20

Entity 8:
- Name: New York City
- Type: GEO
- Description: New York City is the most populous city in the United States
- Degree: 25

Output:
("merge_group"{tuple_delimiter}Apple Inc{graph_field_sep}Apple{tuple_delimiter}0.88{tuple_delimiter}Both entities refer to the exact same technology company - Apple Inc is the official name while Apple is the commonly used short form{tuple_delimiter}Apple Inc{tuple_delimiter}ORGANIZATION){record_delimiter}
("merge_group"{tuple_delimiter}John Smith{graph_field_sep}J. Smith{tuple_delimiter}0.85{tuple_delimiter}Both entities refer to the exact same person - John Smith working as a software engineer at Apple, with J. Smith being the abbreviated name form{tuple_delimiter}John Smith{tuple_delimiter}PERSON){record_delimiter}
("merge_group"{tuple_delimiter}한국생태농업학보{graph_field_sep}Korean Journal of Eco-Agriculture{tuple_delimiter}0.92{tuple_delimiter}These entities are the Korean and English names for the exact same academic journal. The descriptions confirm they refer to the same publication{tuple_delimiter}한국생태농업학보{tuple_delimiter}ORGANIZATION){record_delimiter}
("merge_group"{tuple_delimiter}New York City{graph_field_sep}NYC{tuple_delimiter}0.93{tuple_delimiter}Both entities refer to the exact same city - New York City is the full official name while NYC is the widely used abbreviation{tuple_delimiter}New York City{tuple_delimiter}GEO){completion_delimiter}

#############################
---Real Data---
######################
---Entities to Analyze---
{entities_list}

---Output---"""

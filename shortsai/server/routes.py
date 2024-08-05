import os
from flask import Blueprint, request, render_template
from shortsai.utils import return_base_dir

TEMPLATES_LOCATION =os.path.join(return_base_dir(), "public/templates")

router =  Blueprint(name="routes", import_name="routes", template_folder=TEMPLATES_LOCATION)

@app.post("/generate")
def index_page():
    return render_template("generate.html")
    # prompt = request.form.get("prompt", None)
    # entities = extract_entities(text=prompt)
    # tokens = [wiki_search(entity) for entity in entities]
    # tokens = list(itertools.chain(*tokens))
    # indexes = []
    # for token in tokens:
    #     logger.info("Checking existing redis index")
    #     if not check_existing_redis_index(token):
    #         logger.info("Creating new redis index: %s", token)
    #         logger.info("Getting page content")
    #         page = get_page_content(token)
    #         logger.info("Chunking and saving text")
    #         chunk = chunk_and_save(page)
    #         if not chunk:
    #             continue
    #     indexes.append(token)
    # logger.info("Done checking and creating redis indexes")

    # documents = return_documents(
    #     prompt,
    #     index_names=indexes,
    # )
    # checked_prompt = check_user_prompt(text=prompt, valid_documents=documents)
    # if not checked_prompt:
    #     print("mate, this never happened or I am to old to remember 🥲")
    #     return
    # story = generate_story(user_prompt=prompt, context_documents=documents)
    # print(story.text)


@app.post("story/video_id={video_id}")
def generate_new_video(video_id):
    return render_template("generate.html")
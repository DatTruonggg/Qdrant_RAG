import logging

class Loader():
    @staticmethod
    def load_pdf(data_path):
        import os
        from langchain_community.document_loaders import PyPDFLoader
        if os.path.splitext(data_path)[-1] == ".pdf": # checking if it is not a pdf file
            loader = PyPDFLoader(data_path)
            pages = loader.load_and_split()
            '''
            DON'T NEED: text_list = [page.page_content for page in pages]
            In Qdrant of Langchain framework, the function `from_document` will get the page content.
            '''
            return pages
        else:
            logging.warning("Not a PDF file!!!")
            return None
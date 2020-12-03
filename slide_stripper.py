import pdfplumber

def convert_pdf(file):
    text_string = ""
    with pdfplumber.open(file) as pdf:
        first_page = pdf.pages[0]
        for e in first_page.extract_words():
            # print(e["text"])
            text_string += " " + e["text"]
        string_strip = ""
        string_strip = "".join([i for i in text_string if not i.isdigit()])
        string_strip = string_strip.replace(",", "").replace("  ", "").replace("*","").replace("/", " ")
        print(string_strip)
    return string_strip

if __name__ == "__main__":
    convert_pdf("background-checks.pdf")
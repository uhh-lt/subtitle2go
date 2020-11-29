import pdfplumber


with pdfplumber.open("slides.pdf") as pdf:
    first_page = pdf.pages[12]
    text = first_page.extract_text(x_tolerance=1)
    mytext = text.split()
    for word in mytext:
        if not word.isalnum():
            mytext.remove(word) 
    print(mytext)

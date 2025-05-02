from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, txt="Artificial Intelligence is transforming industries. "
                          "Machine Learning and Natural Language Processing are two key areas.")
pdf.output("sample_test.pdf")

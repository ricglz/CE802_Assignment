all: pilot report
	@echo 'All done!'

pilot:
	pandoc './CE802_Pilot.md' -o './CE802_Pilot.pdf' --template eisvogel --listings --pdf-engine=tectonic

report:
	pandoc './CE802_Report.md' -o './CE802_Report.pdf' --template eisvogel --listings --pdf-engine=tectonic --filter pandoc-fignos

report-tex:
	pandoc './CE802_Report.md' -o './CE802_Report.tex' --template eisvogel --listings --filter pandoc-fignos

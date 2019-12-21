import webbrowser
def get_result_into_html(img_path_and_decode_list):
    html_header = '<!DOCTYPE html>\
    <html lang="en">\
    <head>\
    <title>Bootstrap Example</title>\
    <meta charset="utf-8">\
    <meta name="viewport" content="width=device-width, initial-scale=1">\
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">\
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>\
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>\
    </head>\
    <body>\
    <div class="container">\
    <div class="row">\
        <div class="col-sm-12">\
            <span style="text-align: center;"><h2>Image OCR Decoded Result</h2></span>\
            <table class="table">\
                <thead>\
                <tr>\
                <th scope="col">ID</th>\
                <th scope="col">Image Path </th>\
                <th scope="col">Input Image</th>\
                <th scope="col">Decoded Result</th>\
                </tr>\
                </thead>\
                <tbody>'


    total_text = ""
    for i in range(len(img_path_and_decode_list)):
        text='<tr>\
        <th scope="row">'+str(i)+'</th>\
        <td>'+str(img_path_and_decode_list[i][0])+'</td>\
        <td style ="background:#eee"><img src="'+str(img_path_and_decode_list[i][0])+'"></td>\
        <td>'+str(img_path_and_decode_list[i][1][0])+'</td>\
        </tr>'
        total_text = total_text+text

    text_end_section = '  </tbody></table></div>'

    update_section = html_header+total_text+text_end_section




    html_footer = '</div>\
                </div>\
            </body>\
        </html>'



    html_page = update_section+html_footer


    file_path = "index.html"
    html_file_save =open(file_path,"w")
    html_file_save.write(html_page)
    html_file_save.close()
    print("File Save DONE!")
   
    webbrowser.open_new_tab(file_path)
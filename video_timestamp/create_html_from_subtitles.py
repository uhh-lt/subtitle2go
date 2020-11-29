# encoding: utf-8
# creates a simple HTML output from a subtitle file
# name the file in 

# pip3 install srt
# pip3 install webvtt-py
import srt, webvtt

# reads the subtitle file and for each subtitle utterance takes the start second and the text
# and constructs the subtitle part of the HTML file where the start second is the id of the element
def parse_subtitles():
    filename = 'reference.vtt'
    f = open(filename)
    html=""
    if filename[-4:] == '.vtt':
        for caption in webvtt.read(filename):
            time = caption.start[-6:-4]
            text = caption.text
            html += '<div class="subtitle-line">\n'
            html += '<a id="' + str(time) + '" href="#">'
            html += text
            html += '</a>\n</div>\n\n<br>\n\n'
    elif filename[-4:] == '.srt': 
        print("reading SRT file")
        subtitle_generator = srt.parse(f.read())
        subtitles = list(subtitle_generator)
        for line in subtitles:
            time = line.start.seconds
            text = line.content
            html += '<div class="subtitle-line">\n'
            html += '<a id="' + str(time) + '" href="#">'
            html += text
            html += '</a>\n</div>\n\n<br>\n\n'
    else:
       print("this script only accepts vtt or srt files")         
    return html

# constructs the full HTML file from the generated HTML and saves it
def construct_html(html):

    f = open("subtitles.html","w")
    header = """<!DOCTYPE html>
<html>
<head>
  <title>Clickable subtitles</title>

  <style>
    .auto {
      height: 300px;
      overflow: auto;
      width: 600px;
      border: 1px solid #000;
      padding: 10px;
    }
  </style>

</head>


<body>

  <video id="video" width="622" controls>
    <source src="video.mp4" type="video/mp4">
    Your browser does not support HTML5 video.
  </video>




  <div id="subtitles" class="auto">
  """
    footer = """</div>

  <script>
    function setCurrentTime(event) {
      var video = document.getElementById("video");
      var idName = event.target.id;
      video.currentTime = parseInt(idName);
    }
    var subtitles = document.getElementById("subtitles");
    subtitles.addEventListener('click', setCurrentTime);
  </script>

</body>

</html>
"""
    f.write(header)
    f.write(html)
    f.write(footer)
    f.close()

html = parse_subtitles()
construct_html(html)


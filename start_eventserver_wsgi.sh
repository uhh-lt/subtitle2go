source subtitle2go_env/bin/activate
. path.sh
gunicorn --workers=1 --threads=8 --bind 0.0.0.0:7500 --worker-class=gthread wsgi:app

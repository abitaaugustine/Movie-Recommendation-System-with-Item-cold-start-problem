from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


class MovieForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired()])
    tagline = StringField('Tagline')
    overview = StringField('Overview')
    genres = StringField('Genres')
    cast = StringField('Cast')
    crew = StringField('Crew')
    keywords = StringField('Keywords')
    submit = SubmitField('Submit')
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 16:03:42 2021

@author: dgnistor
"""

# %% Start - Libraries call
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mp
import itertools
from pandas import json_normalize
import json
from requests import request
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import joblib
import pkg_resources

# %% Class creation to automate data extraction

class MoodSearch():
    '''
    Receives: a mood (str) (E.g. 'Happy','Sad')
    Methods: 
        -First 100 playlists tracks & ids
        -Audio Features for all songs in those first 100 playlists
    
    '''
    def __init__(self,moodstr):
        self.moodstr = moodstr
        self.client_id = 'your_client_id'
        self.client_secret = 'your_client secret'


    def auth_spotify_music(self):
        '''
        Receives: client id and client secret
        Returns: 'sp' - the authentication credentials for Spotify
        
        '''
        manager = SpotifyClientCredentials(self.client_id,self.client_secret)
        sp = spotipy.Spotify(client_credentials_manager=manager)
        return sp


    def fetch_mood_playlists(self):
        '''
        Receives: Spotify authentication and mood str
        Returns: A dataframe compiling the first 100 playlists for the indicated mood
        
        '''
        
        sp = self.auth_spotify_music()
        mood_raw = sp.search(self.moodstr,type='playlist')['playlists']
        mood_pl = mood_raw['items']  

        while mood_raw['next'] and len(mood_pl) <100:
            mood_raw = sp.next(mood_raw)['playlists']
            mood_pl.extend(mood_raw['items'])

        playlists_df = json_normalize(mood_pl)
        return playlists_df

    def fetch_playlist_tracks(self): 
        '''
        Receives: Spotify authentication and df retirned from fetch_mood_playlists()
        Returns: A dataframe compiling track_id and track_name included in the first 100 playlists
        
        '''
        sp = self.auth_spotify_music()
        playlists_df = self.fetch_mood_playlists()
        offset = 0
        tracks = []
        
        for i in playlists_df['id']:
            while True:
                content = sp.playlist_tracks(i, fields=None, limit=100, offset=offset, market=None)
                tracks += content['items']
        
                if content['next'] is not None:
                    offset += 100
                else:
                    break
    
            track_id = []
            track_name = []
        
            for track in tracks:
                if track['track'] is not None:
                    track_id.append(track['track']['id'])
                    track_name.append(track['track']['name'])
    
            playlists_tracks_df = pd.DataFrame({"track_id":track_id, "track_name": track_name})
        return playlists_tracks_df
    
    def fetch_audio_features(self):
        '''
        Receives: Spotify authentication and df returned from fetch_playlist_tracks() method
        Returns: A dataframe compiling the audio features for all tracks, and adding Mood column for classification
        
        '''
        sp = self.auth_spotify_music()
        playlists_tracks_df = self.fetch_playlist_tracks()
        tracksaf = []
        for i in playlists_tracks_df['track_id']:
            if i is not None:
                content = sp.audio_features(i)
                tracksaf += content
            #tracksaf.append(content) 
        tracksaf = list(filter(None,tracksaf))
        tracksaf = json_normalize(tracksaf)
        tracksaf['Mood'] = self.moodstr
        return tracksaf
    
    def fetch_full_df(self):
        ''' 
        Receives: Spotify authentication and df returned from fetch_audio_features() method
        Returns: A dataframe compiling the audio features, mood and adding popularity for each track
        
        '''
        sp = self.auth_spotify_music()
        moods_afdf = self.fetch_audio_features()
        popularity = []
       
        for i in moods_afdf['id']:
            popularity.append(sp.track(i)['popularity'])
           
        moods_afdf['popularity'] = popularity
        return moods_afdf
        
# %% Create Moods DF - Automated looping

# Moods_list: input for user
moods_list = ['Sad','Happy','Calm','Energetic']
moods_dfname = [i + 'Df' for i in moods_list]

objs = [MoodSearch(i) for i in moods_list]
    
moods_df = pd.DataFrame()

for obj in objs:
    df = obj.fetch_full_df()
    moods_df = pd.concat([moods_df,df])
   

#%% Class to search a song by name and artist, returning predicted mood for each of the found matches
#Model trained with four moods [['Sad','Happy','Calm','Energetic']]

class SearchAndPredictTrack():
    def __init__(self,song_name,artist):
        self.song_name = song_name
        self.artist = artist
        self.client_id = 'your_client_id'
        self.client_secret = 'your_client secret'
        self.model = pkg_resources.resource_filename(__name__,"random_forest.joblib")


    def auth_spotify_music(self):
        manager = SpotifyClientCredentials(self.client_id,self.client_secret)
        sp = spotipy.Spotify(client_credentials_manager=manager)
        return sp
    
        
    def get_song_id(self):  
        artist= self.artist
        track= self.song_name
        sp = self.auth_spotify_music()
        results = sp.search(q='artist:' + artist + ' track:' + track, type='track')['tracks']['items']
        results = json_normalize(results)
        ids = results['id']
        return ids
    
    def predict_song_mood(self):
            sp = self.auth_spotify_music()
            trackids = self.get_song_id()
            
            name = []
            artist = []
            popularity = []
            audio_features = []
            
            for i in trackids:
                search = sp.track(i)
                
                name.append(search['name'])
                artist.append(search['artists'][0]['name'])
                popularity.append(search['popularity'])
                
                audio_features += (sp.audio_features(i))
                
            content = json_normalize(audio_features)
            content['popularity'] = popularity
            content['name'] = name
            content['artist'] = artist
            content.drop_duplicates(subset=['artist','name'], inplace=True)
            content_model = content.drop(columns=['id','type','uri','track_href','analysis_url','artist','name'],axis=1)  
            
            loaded_rf = self.model
            prediction = loaded_rf.predict(content_model)
            prediction = prediction.tolist()
            content['prediction'] = prediction
            
            names =content['name'].to_list()
            artists = content['artist'].to_list()
            predictions = content['prediction'].to_list()
            
            
            for i, name in enumerate(content['name']):
                print("{0} by {1} is a {2} song".format(names[i],artists[i],predictions[i]))
              
 
#%% Test class with different examples

Attempt = SearchAndPredictTrack('Perhaps, perhaps', 'Cake')
att_result = Attempt.predict_song_mood()


Attempt2 = SearchAndPredictTrack('Going', 'The Rolling Stones')
att_result = Attempt2.predict_song_mood()
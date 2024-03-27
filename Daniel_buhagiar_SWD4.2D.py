from flask import Flask, render_template, request, redirect
import requests
from bs4 import BeautifulSoup
from datetime import date, datetime
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
import random


app = Flask(__name__)
chosen_option = ''


def read_index():
    with open('user_index.txt', 'r') as index:
        user_index = index.read()
    return user_index


def update_index(new_index):
    with open('user_index.txt', 'w') as index:
        index.write(str(new_index))


def webscrape():
    urls = ['https://myanimelist.net/anime/16498/Shingeki_no_Kyojin', 'https://myanimelist.net/anime/1535/Death_Note', 'https://myanimelist.net/anime/5114/Fullmetal_Alchemist__Brotherhood', 'https://myanimelist.net/anime/30276/One_Punch_Man', 'https://myanimelist.net/anime/11757/Sword_Art_Online',
            'https://myanimelist.net/anime/31964/Boku_no_Hero_Academia', 'https://myanimelist.net/anime/38000/Kimetsu_no_Yaiba', 'https://myanimelist.net/anime/20/Naruto', 'https://myanimelist.net/anime/22319/Tokyo_Ghoul', 'https://myanimelist.net/anime/11061/Hunter_x_Hunter_2011']
    synopses = ''
    titles = ''
    for url in urls:
        # synopses
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        synopsis_div = soup.find('div', {'class': 'js-scrollfix-bottom-rel'})
        synopsis_p = synopsis_div.find('p', {'itemprop': 'description'})
        synopsis_text = synopsis_p.text.strip()
        synopses += synopsis_text+'~'

        # titles
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title_h1 = soup.find('h1', {'class': 'title-name h1_bold_none'})
        title_text = title_h1.text.strip()
        titles += title_text+'~'

    with open('synopses.txt', 'w') as synopses_file:
        synopses_file.write(synopses)
    with open('titles.txt', 'w') as titles_file:
        titles_file.write(titles)


def do_i_webscrape():
    with open('last_update.txt', 'r') as update_file:
        last_update = update_file.read()
    if (last_update == ''):
        webscrape()
        with open('last_update.txt', 'w') as update_file:
            update_file.write(str(date.today()))
    else:
        date_format = '%Y-%m-%d'
        last_update_string = datetime.strptime(last_update, date_format).date()
        date_diff = date.today()-last_update_string
        if (date_diff.days >= 14):
            webscrape()
            with open('last_update.txt', 'w') as update_file:
                update_file.write(str(date.today()))


def preprocess(text):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = []
    for word in tokens:
        if word not in stop_words:
            lemmatized_tokens.append(lemmatizer.lemmatize(word))

    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text


def cosine_similarity(first_row, second_row):
    vector_a = np.array(first_row)
    vector_b = np.array(second_row)
    norm_a = norm(vector_a)
    norm_b = norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosine_val = np.dot(vector_a, vector_b) / (norm_a * norm_b)
    return cosine_val


def load_information():
    # pop used to remove the last item since theres an extra ~

    with open('synopses.txt', 'r') as synopses_file:
        text_in_file = synopses_file.read()
        synopses = text_in_file.split('~')
        synopses.pop()

    with open('titles.txt', 'r') as titles_file:
        text_in_file = titles_file.read()
        titles = text_in_file.split('~')
        titles.pop()

    return synopses, titles


def load_scores():
    with open('scores.txt', 'r') as scores_file:
        # Split the string into individual lines
        lines = scores_file.read().strip().split('\n')
        scores_array = []
        for line in lines:
            row = []
            for x in line.strip('[]').split(','):
                value = x.strip()
                if value != "'N/A'":
                    row.append(int(value))
                else:
                    # Removeing the quotes around 'N/A'
                    row.append(value[1:-1])
            scores_array.append(row)
        return scores_array


def load_average_scores():
    with open('average_scores.txt', 'r') as average_scores_file:
        line = average_scores_file.read().strip()
        average_scores_list = []
        average_scores = line.strip('[]').split(',')
        for average_score in average_scores:
            value = average_score.strip()
            if value != "'N/A'":
                average_scores_list.append(float(value))
            else:
                average_scores_list.append(value[1:-1])
        return average_scores_list


def update_average_scores(new_average, anime_index):
    average_scores_list = load_average_scores()
    average_scores_list[anime_index] = new_average

    with open('average_scores.txt', 'w') as average_scores_file:
        average_scores_file.write(str(average_scores_list))


def update_scores(new_score, chosen_option):
    scores = load_scores()
    print(scores)
    # synopses loaded since load_information returns 2 variables
    synopses, titles = load_information()
    if (new_score != 'N/A'):
        scores[int(read_index())][titles.index(chosen_option)] = int(new_score)
    else:
        scores[int(read_index())][titles.index(chosen_option)] = new_score

    # Write the updated scores back to the file
    with open('scores.txt', 'w') as scores_file:
        for row in scores:
            scores_file.write(str(row) + '\n')


def measure_similarity(search, synopses, titles):

    # Will  be used to store and return the final anime titles
    anime_results = []

    synopses_x_titles = synopses + titles

    # To make sure the casing doesnt make a difference
    lower_synopses_x_titles = []
    for item in synopses_x_titles:
        lower_synopses_x_titles.append(item.lower())

    vectorizer = TfidfVectorizer(preprocessor=preprocess)
    tf_idf_matrix = vectorizer.fit_transform(lower_synopses_x_titles)

    search_matrix = vectorizer.transform([search])

    if np.count_nonzero(search_matrix.toarray()[0]) == 0:
        print("Sadly the anime you are looking for cannot be found :(")
    else:
        # The top 5 indexes/similarities are being saved and filtered in case 2 duplicates are in my list,
        # so that the runner up in 5th place can take the final place instead of generating a random one.
        # Of course if there is 1 duplicate and 1 good one the 4th place one will be taken and the fifth will
        # be discarded.
        similarities = [0, 0, 0, 0, 0]
        indexes = [0, 0, 0, 0, 0]

        for i in range(tf_idf_matrix.shape[0]):
            similarity = cosine_similarity(search_matrix.toarray()[0], tf_idf_matrix.toarray()[i])
            print(similarity)
            for index in range(5):
                if similarity > similarities[index]:
                    # Basically makes use of splicing to move everything one position backwords.
                    similarities[index+1:] = similarities[index:4]
                    similarities[index] = similarity
                    indexes[index+1:] = indexes[index:4]
                    indexes[index] = i
                    break

        # Just to note most of the time if it displays Shingeki no Kyojin its because the value is 0 since the default index is 0 therefore it will be changed during processing
        print('Before processing: ', titles[indexes[0] - len(synopses)], titles[indexes[1] - len(synopses)],titles[indexes[2] - len(synopses)], titles[indexes[3] - len(synopses)], titles[indexes[4] - len(synopses)])

        # remove the ones with 0 similarity from lists
        for similarity in similarities[:]:
            if (similarity == 0):
                indexes.pop(similarities.index(similarity))
                similarities.remove(similarity)

        # remove the same animes and keep the ones which have the highest simularity score in the lists
        indexes_to_delete = []
        for index_1 in indexes[:]:
            for index_2 in indexes[:]:
                if index_2 == index_1:
                    continue  # skip the current index
                else:
                    if (titles[index_1-len(synopses)] == titles[index_2-len(synopses)] and similarities[indexes.index(index_1)] > similarities[indexes.index(index_2)]):
                        indexes_to_delete.append(indexes.index(index_2))
        for index in indexes_to_delete[:]:
            indexes.pop(index)
            similarities.pop(index)

        # started using anime_results array at the end and to make it a bit easier to index and manage
        # and to add the acctual result names.
        for index in indexes[:]:
            anime_results.append(titles[index-len(synopses)])

        # if there are 4 or 5 items in the list they are removed
        if len(anime_results) >= 4:
            anime_results = anime_results[:3]

        # adds random titles if the length is not 3 also makes sure that no duplicates are added
        while len(anime_results) < 3:
            random_title = titles[random.randint(0, 9)]
            if random_title not in anime_results:
                anime_results.append(random_title)

        print('After processing: ', anime_results)

    return anime_results


def calculate_average(scores, titles, chosen_option):
    sum_first_items = 0
    count = 0
    for sublist in scores:
        if (sublist[titles.index(chosen_option)] != 'N/A'):
            sum_first_items += int(sublist[titles.index(chosen_option)])
            count += 1
    if (count != 0):
        update_average_scores(round(sum_first_items / count, 2), titles.index(chosen_option))
        return load_average_scores()[titles.index(chosen_option)]
    else:
        return 'N/A'


def read_usernames():
    with open('usernames.txt', 'r') as usernames_file:
        usernames = usernames_file.read()
        usernames = usernames.split('\n')
    return usernames


def read_passwords():
    with open('passwords.txt', 'r') as passwords_file:
        passwords = passwords_file.read()
        passwords = passwords.split('\n')
    return passwords


def remove_username():
    usernames = read_usernames()
    index = int(read_index())
    usernames.pop(index)
    with open('usernames.txt', 'w') as usernames_file:
        username_strings = []
        for username in usernames:
            username_strings.append(str(username))
        usernames_file.write('\n'.join(username_strings))


def remove_password():
    passwords = read_passwords()
    index = int(read_index())
    passwords.pop(index)
    with open('passwords.txt', 'w') as passwords_file:
        password_strings = []
        for password in passwords:
            password_strings.append(str(password))
        passwords_file.write('\n'.join(password_strings))


def remove_scores():
    scores = load_scores()
    scores.pop(int(read_index()))
    with open('scores.txt', 'w') as scores_file:
        for score_row in scores:
            scores_file.write(str(score_row) + '\n')


def change_username(new_username):
    usernames = read_usernames()
    usernames[int(read_index())] = new_username
    with open('usernames.txt', 'w') as usernames_file:
        username_strings = []
        for username in usernames:
            username_strings.append(str(username))
        usernames_file.write('\n'.join(username_strings))


def change_password(new_password):
    passwords = read_passwords()
    passwords[int(read_index())] = new_password
    with open('passwords.txt', 'w') as passwords_file:
        password_strings = []
        for password in passwords:
            password_strings.append(str(password))
        passwords_file.write('\n'.join(password_strings))


def login(username, password):
    if (username != '' and password != ''):
        usernames = read_usernames()
        passwords = read_passwords()
        correct_username = False
        for name in usernames:
            if (name == username):
                correct_username = True
        if (correct_username):
            for passcode in passwords:
                if (passcode == password and passwords.index(password) == usernames.index(username)):
                    user_index = passwords.index(passcode)
                    update_index(user_index)
                    return 'logedin'
        return 'error'
    return 'error'


@app.route('/', methods=['GET', 'POST'])
def index():
    do_i_webscrape()
    user_index = read_index()
    if (user_index == ''):
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            if (login(username, password) == 'logedin'):
                return render_template('search.html', username=read_usernames()[int(read_index())])
            else:
                return render_template('index.html', error='Incorrect Username or Password!')
        return render_template('index.html')
    else:
        return render_template('search.html', username=read_usernames()[int(read_index())])


@app.route('/create_acc', methods=['GET', 'POST'])
def create_acc():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if (username != '' and password != ''):
            usernames = read_usernames()
            for name in usernames:
                if (name == username):
                    return render_template('create_acc.html', error='Username Taken!')
            with open('usernames.txt', 'a') as usernames_file:
                usernames_file.write(str(username)+'\n')
            with open("passwords.txt", "a") as passwords_file:
                passwords_file.write(str(password)+'\n')
            with open("scores.txt", "a") as scores_file:
                scores = []
                for i in range(10):
                    scores.append('N/A')
                scores_file.write(str(scores)+'\n')
            login(username, password)
            if (login(username, password) == 'logedin'):
                usernames = read_usernames()
                return render_template('search.html', username=read_usernames()[int(read_index())])
            else:
                return render_template('index.html', error='Incorrect Username or Password!')
        return render_template('create_acc.html', error='Invalid username or password!')
    return render_template('create_acc.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        chosen = request.form['action']
        if (chosen == '🔎︎ Search:'):
            search = request.form['search']
            lower_search = search.lower()
            synopses, titles = load_information()
            results = measure_similarity(lower_search, synopses, titles)
            if (len(results) == 0):
                return render_template('results.html', search=search, error="Sadly the anime you are looking for cannot be found :( Maybe try being more specific or looking at titles.txt to see whats in our database.", username=read_usernames()[int(read_index())])
            else:
                return render_template('results.html', search=search, results=results, error="If the anime you are searching for is not here try being more specific or looking at titles.txt to see whats in our database.", username=read_usernames()[int(read_index())])
        elif (chosen == 'Profile'):
            return render_template('profile.html', username=read_usernames()[int(read_index())])
        else:
            update_index('')
            return redirect('/')
    else:
        return render_template('search.html', username=read_usernames()[int(read_index())])


@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        global chosen_option
        chosen_option = request.form['action']
        if (chosen_option == 'Sign out'):
            update_index('')
            return redirect('/')
        elif (chosen_option == 'Profile'):
            return render_template('profile.html', username=read_usernames()[int(read_index())])
        elif (chosen_option == '🔎︎ Search'):
            return render_template('search.html', username=read_usernames()[int(read_index())])
        else:
            synopses, titles = load_information()
            scores = load_scores()
            title_index = titles.index(chosen_option)
            return render_template('anime_info.html', title=titles[title_index], synopsis=synopses[title_index], average=calculate_average(scores, titles, chosen_option), score=scores[int(read_index())][title_index], username=read_usernames()[int(read_index())])
    else:
        return render_template('results.html', username=read_usernames()[int(read_index())])


@app.route('/anime_info', methods=['GET', 'POST'])
def anime_info():
    if request.method == 'POST':
        chosen = request.form['action']
        if (chosen == 'Sign out'):
            update_index('')
            return redirect('/')
        elif (chosen == 'Profile'):
            return render_template('profile.html', username=read_usernames()[int(read_index())])
        elif (chosen == '🔎︎ Search'):
            return render_template('search.html', username=read_usernames()[int(read_index())])
        else:
            new_score = request.form['new_score']
            global chosen_option
            update_scores(new_score, chosen_option)
            synopses, titles = load_information()
            scores = load_scores()
            title_index = titles.index(chosen_option)
            return render_template('anime_info.html', title=titles[title_index], synopsis=synopses[title_index], average=calculate_average(scores, titles, chosen_option), score=scores[int(read_index())][title_index], username=read_usernames()[int(read_index())])
    else:
        return render_template('anime_info.html', username=read_usernames()[int(read_index())])


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'POST':
        chosen = request.form['action']
        if (chosen == 'Sign out'):
            update_index('')
            return redirect('/')
        elif (chosen == '🔎︎ Search'):
            return render_template('search.html', username=read_usernames()[int(read_index())])
        elif (chosen == 'Delete account'):
            remove_username()
            remove_password()
            remove_scores()
            update_index('')
            return redirect('/')
        elif (chosen == 'Change Username'):
            if (request.form['current_password'] == read_passwords()[int(read_index())]):
                usernames = read_usernames()
                for name in usernames:
                    if (name == request.form['new_detail']):
                        return render_template('profile.html', message='Username Taken', username=read_usernames()[int(read_index())])
                change_username(request.form['new_detail'])
                return render_template('profile.html', message='Username has been changed', username=read_usernames()[int(read_index())])
            return render_template('profile.html', message='Incorrect current password entered', username=read_usernames()[int(read_index())])
        else:
            if (request.form['current_password'] == read_passwords()[int(read_index())]):
                change_password(request.form['new_detail'])
                return render_template('profile.html', message='Password has been changed', username=read_usernames()[int(read_index())])
            return render_template('profile.html', message='Incorrect current password entered', username=read_usernames()[int(read_index())])
    else:
        return render_template('profile.html', username=read_usernames()[int(read_index())])


if __name__ == '__main__':
    app.run(debug=True)

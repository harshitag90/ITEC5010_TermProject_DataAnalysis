import time
import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
import scipy.stats as stats

# Load the Excel file into a DataFrame
df = pd.read_excel("product_list.xlsx")

st.title("Data Analysis of Healthcare Products on E-Commerce Website: Final Project Showcase")
st.markdown(f'<h1 style="color:#00008B;font-size:20px;">{"Submitted By: Harshita Ghushe"}</h1>',unsafe_allow_html=True)
if st.sidebar.checkbox("Show initial dataset"):
    st.header('Healthcare Dataset')
    st.write(df)

# Cleaning the data
print(df['Name'].dtype)
print(df['Asin'].dtype)
print(df['Price'].dtype)
print(df['Rating'].dtype)
print(df['Rating_Num'].dtype)
print(df['Delivery'].dtype)

# Converting types from object
df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
df['Name'] = df['Name'].astype(str)
df['Asin'] = df['Asin'].astype(str)
df['Rating'] = df['Rating'].astype(str)
df['Rating_Num'] = df['Rating_Num'].str.replace(',', '').astype(int)
df['Delivery'] = df['Delivery'].astype(str)

# Cutting the string for Rating to convert to float
df['Rating'] = df['Rating'].astype(str).str[:3].str.replace(',', '')

# Convert the 'Rating' column to float
df['Rating'] = df['Rating'].astype(float)

# Convert '0' values to NaN (missing value)
df['Delivery'].replace('0', None, inplace=True)

# Convert 'Delivery' column to datetime format
df['Delivery'] = pd.to_datetime(df['Delivery'], format='%a, %b %d', errors='coerce')

# Checking for Duplicates
if st.sidebar.checkbox("Remove Duplicates"):
    duplicates = df.duplicated()
    duplicate_rows = df[duplicates]
    st.write(duplicate_rows)

    # Dropping any duplicates
    df.drop_duplicates(subset=['Asin'], keep='first', inplace=True)


# Distribution of healthcare kit prices in the dataset
# Creating a histogram of the Healthcare kit prices
if st.sidebar.checkbox("Healthcare Kit Prices Distribution"):
    st.subheader("Healthcare Kit Prices Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Price'], bins=10, color='red', ax=ax) 
    ax.set_xlabel('Price')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Healthcare Kit Prices')
    st.pyplot(fig)

if st.sidebar.checkbox("Healthcare Kit Price Distribution (Boxplot)"):
    st.subheader("Healthcare Kit Price Distribution (Boxplot)")
    fig, ax = plt.subplots()
    sns.boxplot(df['Price'], color='salmon', ax=ax) 
    ax.set_ylabel('Price')
    ax.set_title('Distribution of Healthcare Kit Prices')
    st.pyplot(fig)

# Correlation between pricing and ratings
# Calculating the Pearson correlation coefficient
if st.sidebar.checkbox("Pearson Correlation between Prices & Ratings"):
    corr, _ = stats.pearsonr(df['Price'], df['Rating'])
    
    # Creating a table for visualization
    corr_data = {
        'Features': ['Price'],
        'Correlation with Rating': [corr]
    }
    corr_df = pd.DataFrame(corr_data)
    
     # Displaying the table with dark headers
    st.markdown(
        f'<style> .stTable thead tr th {{ color: white; background-color: #333; }} </style>',
        unsafe_allow_html=True
    )
   
    st.table(corr_df)

# Analysis on how ratings of healthcare kit vary by prices
if st.sidebar.checkbox("Price Ranges vs Ratings"):
    st.subheader("Price Ranges vs Ratings")
    
    # Defining the price ranges
    price_ranges = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
    colors = ['blue', 'green', 'orange', 'purple', 'red']

    fig, ax = plt.subplots()

    for i, price_range in enumerate(price_ranges):
        x = df[(df['Price'] >= price_range[0]) & (df['Price'] < price_range[1])]['Price']
        y = df[(df['Price'] >= price_range[0]) & (df['Price'] < price_range[1])]['Rating']
        ax.scatter(x, y, color=colors[i], label=f'${price_range[0]}-${price_range[1]}')

    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Rating')
    ax.legend()
    st.pyplot(fig)


# Classification of Healthcare Kits by Rating Categories
# grouping the data by the 'Rating' column and counting the number of occurrences for each unique value
if st.sidebar.checkbox("Healthcare Kit Ratings Count"):
    st.subheader("Healthcare Kit Ratings Count")
    rating_counts = df.groupby('Rating')['Name'].count()
    fig, ax = plt.subplots()
    ax.bar(rating_counts.index, rating_counts.values, color='green') 
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title('Number of Healthcare Kits by Rating')
    st.pyplot(fig)

# Distribution of Healthcare Kits Across Price Categories with color
if st.sidebar.checkbox("Distribution by Price"):
    st.subheader("Healthcare Kit Prices Distribution")
    bins = pd.cut(df['Price'], bins=10)
    healthcare_by_price = bins.value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(range(len(healthcare_by_price)), healthcare_by_price, color='salmon') 
    ax.set_xticks(range(len(healthcare_by_price)))
    ax.set_xticklabels(['${:.2f}-${:.2f}'.format(bin.left, bin.right) for bin in healthcare_by_price.index], rotation=45)
    ax.set_xlabel('Price Range')
    ax.set_ylabel('Number of Healthcare Kits')
    ax.set_title('Distribution of Healthcare Kits by Price')
    st.pyplot(fig)

# Extracting Year, Day, and Date from the 'Delivery' column
df['Delivery'] = pd.to_datetime(df['Delivery'], format='%a, %b %d', errors='coerce')

# Fill missing years with the current year
current_year = pd.Timestamp.now().year
df['Delivery'] = df['Delivery'].apply(lambda x: x.replace(year=current_year) if pd.notnull(x) and x.year == 1900 else x)

# Filter out rows with valid delivery dates
valid_delivery = df[df['Delivery'].notnull()]

# Extracting Year, Day, and Date
valid_delivery['Day'] = valid_delivery['Delivery'].dt.day_name()
valid_delivery['Date'] = valid_delivery['Delivery'].dt.strftime('%b %d')


# Visualizing Delivery by Day
if st.sidebar.checkbox("Delivery by Day"):
    st.subheader("Delivery by Day")
    delivery_by_day = valid_delivery['Day'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    fig, ax = plt.subplots()
    ax.bar(delivery_by_day.index, delivery_by_day.values, color='salmon')
    ax.set_xlabel('Day')
    ax.set_ylabel('Number of Deliveries')
    ax.set_title('Delivery Count by Day')
    st.pyplot(fig)

# Visualizing Delivery by Date
if st.sidebar.checkbox("Delivery by Date"):
    st.subheader("Delivery by Date")
    delivery_by_date = valid_delivery['Date'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(delivery_by_date.index, delivery_by_date.values, marker='o', linestyle='-', color='green')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Deliveries')
    ax.set_title('Delivery Count by Date')
    ax.xaxis.set_tick_params(rotation=45)
    st.pyplot(fig)

# Filtering out rows with valid delivery dates
valid_delivery = df[df['Delivery'].notnull()]

# Extracting Year, Day, and Date for valid deliveries
valid_delivery['Year'] = valid_delivery['Delivery'].dt.year
valid_delivery['Day'] = valid_delivery['Delivery'].dt.day_name()
valid_delivery['Date'] = valid_delivery['Delivery'].dt.strftime('%b %d')

# Filtering out rows with valid delivery dates
valid_delivery = df[df['Delivery'].notnull()]

# Extracting Year, Day, and Date for valid deliveries
valid_delivery['Year'] = valid_delivery['Delivery'].dt.year
valid_delivery['Day'] = valid_delivery['Delivery'].dt.day_name()
valid_delivery['Date'] = valid_delivery['Delivery'].dt.strftime('%b %d')

# Analysing product names based on their delivery date
if st.sidebar.checkbox("Word Frequency in Product Names by Date"):
    st.subheader("Word Frequency in Product Names by Date")

    # Combine all product names into a single string
    all_names = ' '.join(valid_delivery['Name'].astype(str))

    # Generating word frequency for the top 20 common words
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_names)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Analysing product names based on their delivery day
if st.sidebar.checkbox("Word Frequency in Product Names by Day"):
    st.subheader("Word Frequency in Product Names by Day")

    # Combine product names for each day into a single string
    names_by_day = valid_delivery.groupby('Day')['Name'].apply(lambda x: ' '.join(x.astype(str)))

    # Generating word frequency for the top 20 common words by day
    for day, names in names_by_day.items():
        wordcloud_day = WordCloud(width=800, height=400, background_color='white').generate(names)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_day, interpolation='bilinear')
        plt.title(f"Word Cloud for Product Names on {day}")
        plt.axis('off')
        st.pyplot(plt)


# Visualizing Price by Day
if st.sidebar.checkbox("Price by Day"):
    st.subheader("Price by Day")
    price_by_day = valid_delivery.groupby('Day')['Price'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    fig, ax = plt.subplots()
    ax.bar(price_by_day.index, price_by_day.values, color='orange')
    ax.set_xlabel('Day')
    ax.set_ylabel('Average Price')
    ax.set_title('Average Price by Day')
    st.pyplot(fig)

# Visualizing Ratings by Day
if st.sidebar.checkbox("Ratings by Day"):
    st.subheader("Ratings by Day")
    ratings_by_day = valid_delivery.groupby('Day')['Rating'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    fig, ax = plt.subplots()
    ax.bar(ratings_by_day.index, ratings_by_day.values, color='purple')
    ax.set_xlabel('Day')
    ax.set_ylabel('Average Rating')
    ax.set_title('Average Rating by Day')
    st.pyplot(fig)

# Visualizing Price by Date
if st.sidebar.checkbox("Price by Date"):
    st.subheader("Price by Date")
    price_by_date = valid_delivery.groupby('Date')['Price'].mean().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(price_by_date.index, price_by_date.values, marker='o', linestyle='-', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Price')
    ax.set_title('Average Price by Date')
    ax.xaxis.set_tick_params(rotation=45)
    st.pyplot(fig)
    st.pyplot(fig)

# Visualizing Ratings by Date
if st.sidebar.checkbox("Ratings by Date"):
    st.subheader("Ratings by Date")
    ratings_by_date = valid_delivery.groupby('Date')['Rating'].mean().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ratings_by_date.index, ratings_by_date.values, marker='o', linestyle='-', color='brown')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Rating')
    ax.set_title('Average Rating by Date')
    ax.xaxis.set_tick_params(rotation=45)
    st.pyplot(fig)


# splitting the names into individual words
if st.sidebar.checkbox("Calculate word counts"):
    words = df["Name"].str.split(expand=True).stack()

    # counting the frequency of each word
    word_counts = words.value_counts()

    # showing the top 50 most common words
    st.write(word_counts.head(50))

# Creating a new column with 1 if the name contains certain word and 0 if not
df['Emergency'] = df['Name'].str.contains('personal|outdoor|essential|travel|camping|adventure', case=False).astype(int)
df['Trauma'] = df['Name'].str.contains('kits|bag|kit|pouch', case=False).astype(int)
df['Babycare'] = df['Name'].str.contains('baby', case=False).astype(int)


emergency_percent = df['Emergency'].mean() * 100
trauma_percent = df['Trauma'].mean() * 100
babycare_percent = df['Babycare'].mean() * 100

# Displaying the percentages using Streamlit text display
if st.sidebar.checkbox("Show Percentages of multiple Healthcare Kits"):
    st.subheader("Percentages of Different Healthcare Kits")
    
    st.write(f"Emergency: {emergency_percent:.2f}%")
    st.write(f"Trauma: {trauma_percent:.2f}%")
    st.write(f"Babycare: {babycare_percent:.2f}%")

# filtering data where Emergency column is equal to 1
#if st.sidebar.radio("Rating Numbers type"):
emergency_df = df[df['Emergency'] == 1]

# extracting an array of rating numbers
emergency_array = emergency_df['Rating'].values
emergency_array = emergency_array.astype(float)

# filtering data where Trauma column is equal to 1
trauma_df = df[df['Trauma'] == 1]

# extracting an array of rating numbers
trauma_array = trauma_df['Rating'].values
trauma_array = trauma_array.astype(float)

# filtering data where Babycare column is equal to 1
babycare_df = df[df['Babycare'] == 1]

# extracting an array of rating numbers
babycare_array = babycare_df['Rating'].values
babycare_array = babycare_array.astype(float)

print(emergency_array.dtype)
print(trauma_array.dtype)
print(babycare_array.dtype)


if st.sidebar.checkbox("Healthcare Kit Feature Distribution"):
    st.subheader("Healthcare Kit Feature Distribution")
    data = [emergency_array, trauma_array, babycare_array]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    colors = ['skyblue', 'salmon', 'lightgreen']  # Different colors for each feature
    box_props = dict(facecolor='white', edgecolor='black')  # Boxplot properties

    ax1.boxplot(data, patch_artist=True, boxprops=box_props, showmeans=True)
    ax1.set_ylim([2.5, 5.5])
    ax1.set_xticklabels(['Emergency', 'Trauma', 'Babycare'], color='black')  # X-axis labels color
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Rating')
    ax1.set_title('Distribution of Ratings by Feature')

    for patch, color in zip(ax1.patches, colors):
        patch.set_facecolor(color)

    ax2.boxplot(data, patch_artist=True, boxprops=box_props, showmeans=True)
    ax2.set_ylim([3.7, 4.9])
    ax2.set_xticklabels(['Emergency', 'Trauma', 'Babycare'], color='black')  # X-axis labels color
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Rating')
    ax2.set_title('Ratings Distribution (Focused)')

    for patch, color in zip(ax2.patches, colors):
        patch.set_facecolor(color)

    st.pyplot(fig)



if st.sidebar.checkbox("Different Healthcare Kit Ratio"):
    st.subheader("Different Healthcare Kit Ratio")

    # Assuming you have arrays: emergency_array, trauma_array, babycare_array
    weights = [len(emergency_array), len(trauma_array), len(babycare_array)]
    label = ['Emergency', 'Trauma', 'Babycare']

    fig, ax = plt.subplots()
    ax.set_title('Ratio of Different Healthcare Kits')

    # Creating the pie chart
    ax.pie(weights, labels=label, pctdistance=0.8, autopct='%.2f %%')

    st.pyplot(fig)

# Assuming differentiation based on keywords like 'Emergency', 'Trauma', and 'Babycare'
# Calculate average ratings for each product type
emergency_avg_rating = df[df['Name'].str.contains('emergency', case=False)]['Rating'].mean()
trauma_avg_rating = df[df['Name'].str.contains('trauma', case=False)]['Rating'].mean()
babycare_avg_rating = df[df['Name'].str.contains('baby', case=False)]['Rating'].mean()

# Create a bar plot for comparative analysis
if st.sidebar.checkbox("Comparative Analysis"):
    st.subheader("Average Ratings by Product Type")
    fig, ax = plt.subplots()
    product_types = ['Emergency', 'Trauma', 'Babycare']
    avg_ratings = [emergency_avg_rating, trauma_avg_rating, babycare_avg_rating]
    sns.barplot(x=product_types, y=avg_ratings, palette='viridis')
    ax.set_xlabel('Product Type')
    ax.set_ylabel('Average Rating')
    ax.set_title('Average Ratings for Different Product Types')
    st.pyplot(fig)

# Filtering out rows with valid delivery dates for each product type
valid_delivery_emergency = df[(df['Delivery'].notnull()) & (df['Emergency'] == 1)]
valid_delivery_trauma = df[(df['Delivery'].notnull()) & (df['Trauma'] == 1)]
valid_delivery_babycare = df[(df['Delivery'].notnull()) & (df['Babycare'] == 1)]

# Extracting Year, Day, and Date for valid deliveries of each product type
valid_delivery_emergency['Date'] = valid_delivery_emergency['Delivery'].dt.strftime('%b %d')
valid_delivery_trauma['Date'] = valid_delivery_trauma['Delivery'].dt.strftime('%b %d')
valid_delivery_babycare['Date'] = valid_delivery_babycare['Delivery'].dt.strftime('%b %d')

# Calculating average ratings for each product type based on delivery date
avg_ratings_emergency = valid_delivery_emergency.groupby('Date')['Rating'].mean()
avg_ratings_trauma = valid_delivery_trauma.groupby('Date')['Rating'].mean()
avg_ratings_babycare = valid_delivery_babycare.groupby('Date')['Rating'].mean()

# Merging the average ratings for comparative analysis
avg_ratings_comparison = pd.concat([avg_ratings_emergency, avg_ratings_trauma, avg_ratings_babycare], axis=1)
avg_ratings_comparison.columns = ['Emergency', 'Trauma', 'Babycare']

# Plotting the comparative analysis based on delivery date
if st.sidebar.checkbox("Comparative Analysis by Delivery Date"):
    st.subheader("Average Ratings by Product Type and Delivery Date")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    avg_ratings_comparison.plot(kind='line', ax=ax)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Rating')
    ax.set_title('Comparative Analysis of Average Ratings by Delivery Date')
    ax.legend(title='Product Type')
    
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Filtering out rows with valid delivery dates for each product type
valid_delivery_emergency = df[(df['Delivery'].notnull()) & (df['Emergency'] == 1)]
valid_delivery_trauma = df[(df['Delivery'].notnull()) & (df['Trauma'] == 1)]
valid_delivery_babycare = df[(df['Delivery'].notnull()) & (df['Babycare'] == 1)]

# Extracting Year, Day, and Date for valid deliveries of each product type
valid_delivery_emergency['Date'] = valid_delivery_emergency['Delivery'].dt.strftime('%b %d')
valid_delivery_trauma['Date'] = valid_delivery_trauma['Delivery'].dt.strftime('%b %d')
valid_delivery_babycare['Date'] = valid_delivery_babycare['Delivery'].dt.strftime('%b %d')

# Calculating average prices for each product type based on delivery date
avg_prices_emergency = valid_delivery_emergency.groupby('Date')['Price'].mean()
avg_prices_trauma = valid_delivery_trauma.groupby('Date')['Price'].mean()
avg_prices_babycare = valid_delivery_babycare.groupby('Date')['Price'].mean()

# Merging the average prices for comparative analysis
avg_prices_comparison = pd.concat([avg_prices_emergency, avg_prices_trauma, avg_prices_babycare], axis=1)
avg_prices_comparison.columns = ['Emergency', 'Trauma', 'Babycare']

# Plotting the comparative analysis based on delivery date for average prices
if st.sidebar.checkbox("Comparative Analysis of Average Prices by Delivery Date"):
    st.subheader("Average Prices by Product Type and Delivery Date")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    avg_prices_comparison.plot(kind='line', ax=ax)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Price')
    ax.set_title('Comparative Analysis of Average Prices by Delivery Date')
    ax.legend(title='Product Type')
    
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Counting the number of products for each product type based on delivery date
count_emergency = valid_delivery_emergency.groupby('Date')['Name'].count()
count_trauma = valid_delivery_trauma.groupby('Date')['Name'].count()
count_babycare = valid_delivery_babycare.groupby('Date')['Name'].count()

# Merging the counts for comparative analysis
count_comparison = pd.concat([count_emergency, count_trauma, count_babycare], axis=1)
count_comparison.columns = ['Emergency', 'Trauma', 'Babycare']

# Plotting the comparative analysis based on delivery date for counts of products
if st.sidebar.checkbox("Comparative Analysis of Product Counts by Delivery Date"):
    st.subheader("Product Counts by Product Type and Delivery Date")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    count_comparison.plot(kind='line', ax=ax)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Count of Products')
    ax.set_title('Comparative Analysis of Product Counts by Delivery Date')
    ax.legend(title='Product Type')
    
    plt.xticks(rotation=45)
    st.pyplot(fig)




# Analysis product names length
if st.sidebar.checkbox("Product Name Length Analysis"):
    if 'Name' in df.columns:
        st.subheader("Product Name Lengths")
        df['Name_Length'] = df['Name'].apply(lambda x: len(str(x)))

        # Display the distribution of product name lengths
        fig, ax = plt.subplots()
        sns.histplot(df['Name_Length'], bins=20, kde=True, ax=ax)
        ax.set_xlabel('Product Name Length')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Product Name Lengths')
        st.pyplot(fig)
    else:
        st.warning("Name column not found in the dataset.")

# Scatter plot with differentiation based on keywords in product names
if st.sidebar.checkbox("Multivariate Analysis"):
    # Create a copy of the DataFrame for manipulation
    df_copy = df.copy()

    # Define keywords for different product types
    keywords = {
        'Emergency': 'emergency',
        'Trauma': 'trauma',
        'Babycare': 'baby'
    }

    # Add columns to indicate presence of keywords in product names
    for product_type, keyword in keywords.items():
        df_copy[product_type] = df_copy['Name'].str.contains(keyword, case=False).astype(int)

    # Plotting scatter plot with differentiation based on keywords
    st.subheader("Relationship between Price, Rating, and Product Types")
    fig, ax = plt.subplots()
    colors = ['blue', 'green', 'orange']

    for i, (product_type, keyword) in enumerate(keywords.items()):
        subset = df_copy[df_copy[product_type] == 1]
        ax.scatter(subset['Price'], subset['Rating'], label=product_type, color=colors[i], alpha=0.7)

    ax.set_xlabel('Price')
    ax.set_ylabel('Rating')
    ax.set_title('Relationship between Price, Rating, and Product Types')
    ax.legend()
    st.pyplot(fig)


# Word Cloud for common words in product names
if st.sidebar.checkbox("Word Cloud for Product Names"):
    st.subheader("Word Cloud for Product Names")
    
    # Combine all product names into a single string
    all_names = ' '.join(df['Name'].astype(str))

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_names)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Correlation Matrix
if st.sidebar.checkbox("Correlation Matrix"):
    st.subheader("Correlation Matrix")
    
    # Select relevant columns for correlation analysis
    correlation_columns = ['Price', 'Rating', 'Rating_Num']

    # Calculate the correlation matrix
    correlation_matrix = df[correlation_columns].corr()

    # Plot the correlation matrix as a heatmap
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig)

# Distribution of Ratings for each Product Type
if st.sidebar.checkbox("Distribution of Ratings for Each Product Type"):
    st.subheader("Distribution of Ratings for Each Product Type")
    keywords = {
        'Emergency': 'emergency',
        'Trauma': 'trauma',
        'Babycare': 'baby'
    }
    df_copy = df.copy()
    fig, ax = plt.subplots()
    for product_type, keyword in keywords.items():
        subset = df_copy[df_copy[product_type] == 1]
        sns.histplot(subset['Rating'], kde=True, label=product_type, alpha=0.7)

    ax.set_xlabel('Rating')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Ratings for Each Product Type')
    ax.legend()
    st.pyplot(fig)



if st.sidebar.checkbox("Clustering Analysis using K-Means"):
    # Select relevant numerical columns for clustering
    columns_for_clustering = ['Price', 'Rating', 'Rating_Num']  # Adjust these columns as needed

    # Filter the DataFrame to include only numerical columns
    numerical_data = df[columns_for_clustering]

    # Standardize the numerical data for clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_data)

    # Perform K-means clustering
    num_clusters = 3  # Number of clusters to identify
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)

    # Add the cluster labels to the DataFrame
    df['Cluster'] = clusters

    # Display the counts of products in each cluster
    st.subheader("Product Clusters")
    st.write(df['Cluster'].value_counts())

    # Scatterplot of Price vs Rating colored by clusters
    st.subheader("Clustering Analysis: Price vs Rating")
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['Price'], df['Rating'], c=clusters, cmap='viridis')
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.set_xlabel('Price')
    ax.set_ylabel('Rating')
    ax.set_title('K-means Clustering: Price vs Rating')
    st.pyplot(fig)


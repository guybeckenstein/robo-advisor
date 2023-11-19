_Final project for the Fintech workshop by Zevin_
<style>
  h1, h3, h4 {
    text-align: center;
  }
</style>
<body>
  <h1 style="background-color: rgb(255, 100, 100); padding: 2px;">
    robo-advisor
  </h1>
  <!-- Without Docker -->
  <div>
    <h1>
      WITHOUT DOCKER
    </h1>
    <!-- Installation prerequisites -->
    <h3>
      The installations that need to be done in advance:
    </h3>
    <ol type="1">
      <li>
        Git
      </li>
      <li>
        Python 3.11
      </li>
      <li>
        PostgreSQL
      </li>
      <li>
        PyTest
      </li>
    </ol>
    <!-- Configurations -->
    <h4 style="padding-top: 20px;">
      Make these configurations:
    </h4>
    <small style="color: red;">
      * You must configure a database called `roboadvisor` in your postgres! Otherwise, it won't work. Use pgAdmin 4 or PSQL for it.
      <br>
      ** You should also create a file called `.env.dev` in the root folder (robo-advisor), and add environment variables there
    </small>
    <ol type="1">
      <li>
        Create a new Django project and go to `settings.py` that you've just created. Copy `SECRET_KEY` value
      </li>
      <li>
        Create a file in the root folder (robo-advisor) named `.env.dev` (exactly)
      </li>
      <li>
        Add the following lines:
        <br>
        <code>
          DEBUG=1
          <br>
          SECRET_KEY=PUT_HERE_YOUR_PREVIOUS_PROJECT_SECRET_KEY
          <br>
          POSTGRES_HOST=db
          <br>
          POSTGRES_PORT=5432
          <br>
          POSTGRES_DB=roboadvisor
          <br>
          POSTGRES_USER=PUT_HERE_YOUR_LOCAL_POSTGRES_USERNAME
          <br>
          POSTGRES_PASSWORD=PUT_HERE_YOUR_LOCAL_POSTGRES_PASSWORD
          <br>
          CSRF_TRUSTED_ORIGINS=http://0.0.0.0:1337
        </code>
      </li>
      <li>
        Update the values of these environment variables according to you:
        <code>
          PUT_HERE_YOUR_PREVIOUS_PROJECT_SECRET_KEY
        </code>
        <code>
          PUT_HERE_YOUR_LOCAL_POSTGRES_USERNAME
        </code>
        <code>
          PUT_HERE_YOUR_LOCAL_POSTGRES_PASSWORD
        </code>
      </li>
    </ol>
    <!-- Commands -->
    <h3>
      Run these commands:
    </h3>
    <code>
      pip install -r requirements.txt
    </code>
    <br>
    <code>
      python manage.py migrate
    </code>
    <br>
    <code>
      python manage.py runserver
    </code>
    <a href="localhost:8000">
      <h3>
        Now enter this link!
      </h3>
    </a>
  </div>
  <!-- With Docker -->
</body>


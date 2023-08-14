document.addEventListener("DOMContentLoaded", function() {
    // Get the current date
    var currentDate = new Date();

    // Calculate the date 10 years earlier
    var tenYearsAgo = new Date(currentDate);
    tenYearsAgo.setFullYear(currentDate.getFullYear() - 10);

    // Format the date as "YYYY-MM-DD"
    var formattedDate = tenYearsAgo.toISOString().substr(0, 10);

    // Set the value of the date input to the calculated date
    document.getElementById("id_start_date").value = formattedDate;

    // Set the value of the end date input to the current date
    var formattedCurrentDate = currentDate.toISOString().substr(0, 10);
    document.getElementById("id_end_date").value = formattedCurrentDate;
});
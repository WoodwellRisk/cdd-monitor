// https://developer.mozilla.org/en-US/docs/Web/API/Window/matchMedia
// 576px is the mobile breakpoint for our app
let windowSizeMatch = window.matchMedia("(min-width: 576px)");

// https://stackoverflow.com/questions/45947570/how-to-attach-an-event-listener-to-the-dom-depending-upon-the-screen-size/45950288#45950288
// m in the function arg is the matchMedia object, passed back into the function
windowSizeMatch.addEventListener('change', function(m) {
    if (m.matches) {
        let sidebarContainer = document.querySelector('#sidebar-container');
        sidebarContainer.style.display = 'block';
    }
    else {
        let sidebarContainer = document.querySelector('#sidebar-container');
        sidebarContainer.style.display = 'none';
    }
});
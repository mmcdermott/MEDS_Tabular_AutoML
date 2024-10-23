function toggleFolder(folderId) {
  const content = document.getElementById(folderId);
  const folderItem = content.previousElementSibling;

  // Toggle active state on folder item
  folderItem.classList.toggle('active');

  // Toggle visibility of content
  content.classList.toggle('visible');

  // Prevent event bubbling
  event.stopPropagation();
}

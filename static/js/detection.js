$(function () {
    var detection_cmd = ['async', 'sync'];
    var flip_cmd = ['flip-x', 'flip-y', 'flip-xy', 'flip-reset'];
    var url = "";
    $('.btn').on('click', function () {
        var command = JSON.stringify({ "command": $('#' + $(this).attr('id')).val() });
        if (JSON.parse(command).command == "") {
            var command = JSON.stringify({ "command": $(this).find('input').val() });
        }
        //console.log(command)
        if (detection_cmd.includes(JSON.parse(command).command)) {
            url = '/detection';
            post(url, command);
        }
        if (flip_cmd.includes(JSON.parse(command).command)) {
            url = '/detection';
            post(url, command);
        }
    });
    function post(url, command) {
        $.ajax({
            type: 'POST',
            url: url,
            data: command,
            contentType: 'application/json',
            timeout: 10000
        }).done(function (data) {
            var sent_cmd = JSON.parse(command).command;
            var is_async_mode = JSON.parse(data.ResultSet).is_async_mode;
            var flip_code = JSON.parse(data.ResultSet).flip_code;
            $("#res").text("is_async_mode: " + is_async_mode + " / flip_code: " + flip_code);
            if (sent_cmd == 'async') {
                $("#async").attr('class', 'btn btn-danger');
                $("#sync").attr('class', 'btn btn-dark');
            }
            if (sent_cmd == 'sync') {
                $("#sync").attr('class', 'btn btn-danger');
                $("#async").attr('class', 'btn btn-dark');
            }
        }).fail(function (jqXHR, textStatus, errorThrown) {
            $("#res").text(textStatus + ":" + jqXHR.status + " " + errorThrown);
        });
        return false;
    }
    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    });
});

